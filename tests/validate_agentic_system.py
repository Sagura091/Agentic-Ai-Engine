#!/usr/bin/env python3
"""
Agentic AI System Validation Script

This script provides a simple interface to validate that your agentic AI system
is truly functional and not just appearing to work. It runs the comprehensive
backend testing suite and provides clear results.

Usage:
    python tests/validate_agentic_system.py
    python tests/validate_agentic_system.py --quick
    python tests/validate_agentic_system.py --full
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import the comprehensive test runner
try:
    from tests.backend_comprehensive.run_comprehensive_tests import ComprehensiveTestRunner
except ImportError as e:
    print(f"❌ Error importing test modules: {e}")
    print()
    print("🔧 SETUP REQUIRED:")
    print("The comprehensive testing suite needs to be properly installed.")
    print()
    print("Please ensure you have:")
    print("1. All test files in tests/backend_comprehensive/ directory")
    print("2. Required dependencies installed:")
    print("   pip install pytest pytest-asyncio structlog psutil")
    print()
    print("3. Python path configured correctly")
    print("   export PYTHONPATH=\"${PYTHONPATH}:$(pwd)\"")
    print()
    print("If you're running from the project root, try:")
    print("   python -m tests.validate_agentic_system")
    print()
    sys.exit(1)


def print_banner():
    """Print validation banner."""
    print("=" * 80)
    print("🤖 AGENTIC AI SYSTEM VALIDATION")
    print("=" * 80)
    print("This validation suite will test whether you have REAL agentic AI agents")
    print("or just systems that appear to be agentic.")
    print()
    print("Key Validation Areas:")
    print("✓ Agent Creation & Configuration")
    print("✓ LLM Integration & Communication") 
    print("✓ Tool Integration & Usage")
    print("✓ RAG & Knowledge Base Integration")
    print("✓ Autonomous Behavior & Decision Making")
    print("✓ Performance & Scalability")
    print("✓ End-to-End Workflows")
    print("=" * 80)
    print()


def print_results_summary(summary):
    """Print formatted results summary."""
    status_colors = {
        "EXCELLENT": "🟢",
        "GOOD": "🟡", 
        "ACCEPTABLE": "🟠",
        "NEEDS_IMPROVEMENT": "🔴",
        "CRITICAL_ISSUES": "💀"
    }
    
    status_icon = status_colors.get(summary["overall_status"], "❓")
    
    print("=" * 80)
    print("📊 VALIDATION RESULTS")
    print("=" * 80)
    print(f"Overall Status: {status_icon} {summary['overall_status']}")
    print(f"Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"Tests Passed: {summary['total_passed']}/{summary['total_tests']}")
    print(f"Duration: {summary['total_duration']:.1f} seconds")
    print()
    
    # Test breakdown
    print("📋 Test Breakdown:")
    breakdown = summary.get("test_breakdown", {})
    for severity, results in breakdown.items():
        if results["total"] > 0:
            rate = results["passed"] / results["total"]
            icon = "✅" if rate >= 0.8 else "⚠️" if rate >= 0.6 else "❌"
            print(f"  {icon} {severity.title()}: {results['passed']}/{results['total']} ({rate:.1%})")
    
    print()
    
    # Suite results
    print("🧪 Test Suite Results:")
    for suite_name, result in summary["suite_results"].items():
        if "error" in result:
            print(f"  ❌ {suite_name}: ERROR - {result['error']}")
        else:
            rate = result.get("success_rate", 0)
            icon = "✅" if rate >= 0.8 else "⚠️" if rate >= 0.6 else "❌"
            print(f"  {icon} {suite_name}: {result['passed']}/{result['total_tests']} ({rate:.1%})")
    
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    for rec in summary.get("recommendations", []):
        print(f"  • {rec}")
    
    print("=" * 80)


def interpret_results(summary):
    """Interpret and explain results."""
    status = summary["overall_status"]
    success_rate = summary["overall_success_rate"]
    critical_rate = summary.get("critical_tests_passed", 0)
    
    print("🔍 RESULT INTERPRETATION")
    print("=" * 80)
    
    if status == "EXCELLENT":
        print("🎉 CONGRATULATIONS! Your agentic AI system is working excellently!")
        print("   • Agents are properly created and configured")
        print("   • Autonomous behavior is demonstrated")
        print("   • All integrations are functional")
        print("   • System is ready for production use")
        
    elif status == "GOOD":
        print("👍 Your agentic AI system is working well!")
        print("   • Core functionality is solid")
        print("   • Minor issues may need attention")
        print("   • System is suitable for most use cases")
        
    elif status == "ACCEPTABLE":
        print("⚠️  Your agentic AI system has basic functionality")
        print("   • Core agents are working")
        print("   • Some components need improvement")
        print("   • Review failed tests for specific issues")
        
    elif status == "NEEDS_IMPROVEMENT":
        print("🔧 Your agentic AI system needs significant work")
        print("   • Critical functionality may be missing")
        print("   • Agents may not be truly autonomous")
        print("   • Address failing tests before production")
        
    else:  # CRITICAL_ISSUES
        print("🚨 CRITICAL ISSUES DETECTED!")
        print("   • Your system may not have working agentic AI")
        print("   • Agents may be scripted rather than autonomous")
        print("   • Immediate attention required")
    
    print()
    
    # Specific guidance
    if critical_rate < 0.8:
        print("🎯 PRIORITY ACTIONS:")
        print("   1. Fix agent creation and configuration issues")
        print("   2. Ensure LLM integration is working properly")
        print("   3. Validate autonomous behavior patterns")
        print("   4. Test decision-making capabilities")
    
    elif success_rate < 0.7:
        print("🎯 RECOMMENDED ACTIONS:")
        print("   1. Review integration test failures")
        print("   2. Check tool and RAG system functionality")
        print("   3. Optimize performance if needed")
        print("   4. Validate end-to-end workflows")
    
    else:
        print("🎯 NEXT STEPS:")
        print("   1. Monitor system performance in production")
        print("   2. Implement additional custom tests as needed")
        print("   3. Regular validation runs recommended")
    
    print("=" * 80)


async def run_validation(mode="standard"):
    """Run validation based on mode."""
    runner = ComprehensiveTestRunner()
    
    print(f"🚀 Starting {mode} validation...")
    print()
    
    if mode == "quick":
        print("Running critical tests only (fastest validation)...")
        summary = await runner.run_critical_tests_only()
        
    elif mode == "full":
        print("Running complete test suite (includes slow and stress tests)...")
        summary = await runner.run_all_tests(include_slow=True, include_stress=True)
        
    else:  # standard
        print("Running standard test suite...")
        summary = await runner.run_all_tests(include_slow=False, include_stress=False)
    
    return summary


async def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate your agentic AI system functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/validate_agentic_system.py           # Standard validation
  python tests/validate_agentic_system.py --quick   # Quick critical tests only
  python tests/validate_agentic_system.py --full    # Complete validation suite
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Run only critical tests (fastest)"
    )
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="Run complete test suite including slow and stress tests"
    )
    parser.add_argument(
        "--save-results", 
        type=str, 
        help="Save detailed results to file"
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.quick:
        mode = "quick"
    elif args.full:
        mode = "full"
    else:
        mode = "standard"
    
    try:
        print_banner()
        
        # Run validation
        summary = await run_validation(mode)
        
        # Print results
        print_results_summary(summary)
        print()
        interpret_results(summary)
        
        # Save results if requested
        if args.save_results:
            runner = ComprehensiveTestRunner()
            runner.results = summary.get("suite_results", {})
            output_file = runner.save_results(args.save_results)
            print(f"📁 Detailed results saved to: {output_file}")
            print()
        
        # Determine exit code
        if summary["overall_status"] in ["EXCELLENT", "GOOD"]:
            print("✅ Validation completed successfully!")
            return 0
        elif summary["overall_status"] == "ACCEPTABLE":
            print("⚠️  Validation completed with warnings.")
            return 1
        else:
            print("❌ Validation failed - critical issues detected.")
            return 2
            
    except Exception as e:
        print(f"💥 Validation failed with error: {e}")
        print("Please check your system configuration and try again.")
        return 3


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
