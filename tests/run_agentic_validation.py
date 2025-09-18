#!/usr/bin/env python3
"""
Main script for running comprehensive agentic intelligence validation.

This script executes the complete test suite to validate true agentic AI
capabilities and generate detailed reports on agent intelligence.
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.agentic_intelligence_validation import (
    AgenticIntelligenceValidator,
    AgenticCapability,
    AgenticBehaviorMetrics
)
from app.core.seamless_integration import seamless_integration
from app.orchestration.enhanced_orchestrator import enhanced_orchestrator


class AgenticValidationReporter:
    """Generate comprehensive reports for agentic validation results."""
    
    def __init__(self):
        """Initialize the reporter."""
        self.report_timestamp = datetime.now()
    
    def generate_comprehensive_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report."""
        report = []
        
        # Header
        report.append("=" * 80)
        report.append("🧠 COMPREHENSIVE AGENTIC INTELLIGENCE VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Session ID: {validation_results.get('test_session_id', 'N/A')}")
        report.append(f"Agents Tested: {len(validation_results.get('agents_tested', []))}")
        report.append("")
        
        # Executive Summary
        report.append("📊 EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        aggregate_metrics = validation_results.get("aggregate_metrics")
        if aggregate_metrics:
            overall_score = aggregate_metrics.overall_agentic_score
            report.append(f"Overall Agentic Intelligence Score: {overall_score:.2%}")
            
            if overall_score >= 0.8:
                report.append("🎉 VERDICT: HIGHLY AGENTIC - True autonomous intelligence demonstrated")
            elif overall_score >= 0.65:
                report.append("✅ VERDICT: MODERATELY AGENTIC - Clear autonomous behavior with room for improvement")
            elif overall_score >= 0.4:
                report.append("⚠️  VERDICT: LIMITED AGENTIC - Some autonomous behavior, significant scripted responses")
            else:
                report.append("❌ VERDICT: NON-AGENTIC - Primarily scripted responses, minimal autonomy")
        
        report.append("")
        
        # Capability Breakdown
        report.append("🎯 CAPABILITY ANALYSIS")
        report.append("-" * 40)
        
        if aggregate_metrics:
            capabilities = [
                ("Autonomous Decision Making", aggregate_metrics.decision_independence),
                ("Creative Problem Solving", aggregate_metrics.creative_problem_solving),
                ("Adaptive Learning", aggregate_metrics.adaptive_learning_rate),
                ("Goal-Oriented Behavior", aggregate_metrics.goal_persistence),
                ("Intelligent Tool Usage", aggregate_metrics.tool_usage_intelligence),
                ("Emergent Behavior", aggregate_metrics.emergent_behavior_score),
                ("Collaboration Intelligence", aggregate_metrics.collaboration_effectiveness)
            ]
            
            for capability, score in capabilities:
                status = self._get_capability_status(score)
                report.append(f"{capability:.<30} {score:.2%} {status}")
        
        report.append("")
        
        # Individual Agent Results
        report.append("🤖 INDIVIDUAL AGENT ANALYSIS")
        report.append("-" * 40)
        
        individual_results = validation_results.get("individual_results", {})
        classifications = validation_results.get("agentic_classification", {})
        
        for agent_id, agent_results in individual_results.items():
            classification = classifications.get(agent_id, {})
            
            report.append(f"\nAgent: {agent_id}")
            report.append(f"Classification: {classification.get('classification', 'Unknown')}")
            report.append(f"Overall Score: {agent_results.get('overall_score', 0):.2%}")
            report.append(f"Truly Agentic: {'Yes' if agent_results.get('is_truly_agentic', False) else 'No'}")
            
            strengths = classification.get('strengths', [])
            weaknesses = classification.get('weaknesses', [])
            
            if strengths:
                report.append(f"Strengths: {', '.join(strengths)}")
            if weaknesses:
                report.append(f"Weaknesses: {', '.join(weaknesses)}")
        
        report.append("")
        
        # Detailed Test Evidence
        report.append("🔍 DETAILED TEST EVIDENCE")
        report.append("-" * 40)
        
        for agent_id, agent_results in individual_results.items():
            report.append(f"\n--- Agent {agent_id} ---")
            
            for test_result in agent_results.get("test_results", []):
                report.append(f"\nTest: {test_result.test_name}")
                report.append(f"Capability: {test_result.capability.value}")
                report.append(f"Score: {test_result.score:.2%}")
                report.append(f"Agentic: {'Yes' if test_result.is_truly_agentic else 'No'}")
                report.append(f"Reasoning: {test_result.reasoning}")
        
        # Collaboration Analysis
        if "collaboration_result" in validation_results:
            collab_result = validation_results["collaboration_result"]
            report.append(f"\n🤝 COLLABORATION ANALYSIS")
            report.append("-" * 40)
            report.append(f"Collaboration Score: {collab_result.score:.2%}")
            report.append(f"Truly Collaborative: {'Yes' if collab_result.is_truly_agentic else 'No'}")
            report.append(f"Analysis: {collab_result.reasoning}")
        
        report.append("")
        
        # Recommendations
        recommendations = validation_results.get("recommendations", [])
        if recommendations:
            report.append("💡 RECOMMENDATIONS")
            report.append("-" * 40)
            for i, recommendation in enumerate(recommendations, 1):
                report.append(f"{i}. {recommendation}")
        
        report.append("")
        
        # Comparison with Baseline
        report.append("📈 BASELINE COMPARISON")
        report.append("-" * 40)
        report.append("Comparison with typical non-agentic systems:")
        
        if aggregate_metrics:
            baseline_comparison = [
                ("Decision Independence", aggregate_metrics.decision_independence, 0.3),
                ("Creative Problem Solving", aggregate_metrics.creative_problem_solving, 0.2),
                ("Adaptive Learning", aggregate_metrics.adaptive_learning_rate, 0.1),
                ("Goal Persistence", aggregate_metrics.goal_persistence, 0.4),
                ("Tool Intelligence", aggregate_metrics.tool_usage_intelligence, 0.3),
                ("Emergent Behavior", aggregate_metrics.emergent_behavior_score, 0.1),
                ("Collaboration", aggregate_metrics.collaboration_effectiveness, 0.2)
            ]
            
            for capability, score, baseline in baseline_comparison:
                improvement = ((score - baseline) / baseline * 100) if baseline > 0 else 0
                report.append(f"{capability:.<25} {score:.2%} vs {baseline:.2%} ({improvement:+.0f}%)")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _get_capability_status(self, score: float) -> str:
        """Get status indicator for capability score."""
        if score >= 0.8:
            return "🟢 Excellent"
        elif score >= 0.65:
            return "🟡 Good"
        elif score >= 0.4:
            return "🟠 Limited"
        else:
            return "🔴 Poor"
    
    def save_detailed_json_report(self, validation_results: Dict[str, Any], filename: str) -> None:
        """Save detailed results as JSON."""
        # Convert datetime objects to strings for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        with open(filename, 'w') as f:
            json.dump(validation_results, f, indent=2, default=json_serializer)


async def main():
    """Main function to run comprehensive agentic validation."""
    print("🧠 Starting Comprehensive Agentic Intelligence Validation")
    print("=" * 70)
    
    try:
        # Initialize validator
        validator = AgenticIntelligenceValidator()
        await validator.initialize_validation_environment()
        
        print("✅ Validation environment initialized")
        
        # Get available agents for testing
        available_agents = list(enhanced_orchestrator.agents.keys())
        
        if not available_agents:
            print("⚠️  No agents available. Creating test agents...")
            
            # Create test agents
            test_agents = []
            agent_types = ["autonomous", "research", "creative", "optimization"]
            
            for agent_type in agent_types:
                agent_id = await seamless_integration.create_unlimited_agent(
                    agent_type=agent_type,
                    name=f"Test {agent_type.title()} Agent",
                    description=f"Agent for testing {agent_type} capabilities"
                )
                test_agents.append(agent_id)
                print(f"✅ Created test agent: {agent_type} ({agent_id})")
            
            available_agents = test_agents
        
        print(f"🤖 Testing {len(available_agents)} agents")
        
        # Run comprehensive validation
        print("\n🔍 Running comprehensive agentic validation...")
        validation_results = await validator.run_comprehensive_agentic_validation(available_agents)
        
        # Generate reports
        reporter = AgenticValidationReporter()
        
        # Generate and display comprehensive report
        comprehensive_report = reporter.generate_comprehensive_report(validation_results)
        print("\n" + comprehensive_report)
        
        # Save detailed JSON report
        json_filename = f"agentic_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        reporter.save_detailed_json_report(validation_results, json_filename)
        print(f"\n📄 Detailed JSON report saved: {json_filename}")
        
        # Save text report
        text_filename = f"agentic_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_filename, 'w') as f:
            f.write(comprehensive_report)
        print(f"📄 Text report saved: {text_filename}")
        
        # Final verdict
        aggregate_metrics = validation_results.get("aggregate_metrics")
        if aggregate_metrics:
            overall_score = aggregate_metrics.overall_agentic_score
            
            print(f"\n🎯 FINAL VERDICT")
            print("=" * 30)
            print(f"Overall Agentic Intelligence Score: {overall_score:.2%}")
            
            if overall_score >= 0.8:
                print("🎉 RESULT: TRUE AGENTIC AI CONFIRMED!")
                print("   Your agents demonstrate genuine autonomous intelligence.")
            elif overall_score >= 0.65:
                print("✅ RESULT: AGENTIC BEHAVIOR DETECTED")
                print("   Your agents show clear autonomous capabilities with room for improvement.")
            else:
                print("⚠️  RESULT: LIMITED AGENTIC BEHAVIOR")
                print("   Your agents need significant improvement to achieve true autonomy.")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
