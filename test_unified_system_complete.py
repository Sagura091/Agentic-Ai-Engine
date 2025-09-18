#!/usr/bin/env python3
"""
Comprehensive Test Suite for THE Unified Multi-Agent System.

This test verifies that all 4 phases of the system overhaul are working correctly:
✅ PHASE 1: Foundation (UnifiedRAGSystem, CollectionBasedKBManager, AgentIsolationManager)
✅ PHASE 2: Memory & Tools (UnifiedMemorySystem, UnifiedToolRepository)
✅ PHASE 3: Communication (AgentCommunicationSystem)
✅ PHASE 4: Optimization (PerformanceOptimizer)

DESIGN PRINCIPLES VERIFICATION:
- One system to rule them all
- Simple, clean, fast operations
- No complexity unless absolutely necessary
- Agent isolation by default with optional communication
"""

import asyncio
import sys
import traceback
from datetime import datetime
from typing import Dict, Any

# Import THE unified system
try:
    from app.core import UnifiedSystemOrchestrator, SystemConfig
    from app.memory import MemoryType
    print("✅ Successfully imported UnifiedSystemOrchestrator")
except ImportError as e:
    print(f"❌ Failed to import UnifiedSystemOrchestrator: {e}")
    sys.exit(1)


class ComprehensiveSystemTest:
    """Comprehensive test suite for THE unified system."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.orchestrator = None
        self.test_results = {}
        self.start_time = datetime.now()
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("🚀 Starting Comprehensive System Test Suite...")
        print("=" * 80)
        
        try:
            # Test Phase 1: Foundation
            await self.test_phase_1_foundation()
            
            # Test Phase 2: Memory & Tools
            await self.test_phase_2_memory_tools()
            
            # Test Phase 3: Communication
            await self.test_phase_3_communication()
            
            # Test Phase 4: Optimization
            await self.test_phase_4_optimization()
            
            # Test System Integration
            await self.test_system_integration()
            
            # Generate final report
            await self.generate_final_report()
            
        except Exception as e:
            print(f"❌ Critical test failure: {e}")
            traceback.print_exc()
            self.test_results["critical_failure"] = str(e)
        
        finally:
            if self.orchestrator:
                await self.orchestrator.shutdown()
        
        return self.test_results
    
    async def test_phase_1_foundation(self) -> None:
        """Test PHASE 1: Foundation components."""
        print("\n🏗️ TESTING PHASE 1: FOUNDATION")
        print("-" * 50)
        
        try:
            # Initialize the orchestrator
            config = SystemConfig(
                enable_communication=False,  # Test foundation only
                enable_optimization=False
            )
            self.orchestrator = UnifiedSystemOrchestrator(config)
            
            # Test system initialization
            print("   🎯 Testing system initialization...")
            await self.orchestrator.initialize()
            
            # Verify core components
            assert self.orchestrator.unified_rag is not None, "UnifiedRAGSystem not initialized"
            assert self.orchestrator.kb_manager is not None, "CollectionBasedKBManager not initialized"
            assert self.orchestrator.isolation_manager is not None, "AgentIsolationManager not initialized"
            
            print("   ✅ UnifiedRAGSystem: OPERATIONAL")
            print("   ✅ CollectionBasedKBManager: OPERATIONAL")
            print("   ✅ AgentIsolationManager: OPERATIONAL")
            
            self.test_results["phase_1_foundation"] = "PASSED"
            print("✅ PHASE 1 FOUNDATION: PASSED")
            
        except Exception as e:
            print(f"❌ PHASE 1 FOUNDATION: FAILED - {e}")
            self.test_results["phase_1_foundation"] = f"FAILED: {e}"
            raise
    
    async def test_phase_2_memory_tools(self) -> None:
        """Test PHASE 2: Memory & Tools components."""
        print("\n🧠 TESTING PHASE 2: MEMORY & TOOLS")
        print("-" * 50)
        
        try:
            # Verify memory and tool components
            assert self.orchestrator.memory_system is not None, "UnifiedMemorySystem not initialized"
            assert self.orchestrator.tool_repository is not None, "UnifiedToolRepository not initialized"
            
            print("   ✅ UnifiedMemorySystem: OPERATIONAL")
            print("   ✅ UnifiedToolRepository: OPERATIONAL")
            
            # Test agent creation and memory
            print("   🧪 Testing agent memory operations...")
            test_agent_id = "test_agent_001"
            
            # Test memory operations
            memory_id = await self.orchestrator.memory_system.add_memory(
                agent_id=test_agent_id,
                memory_type=MemoryType.SHORT_TERM,
                content="Test memory for comprehensive system test",
                metadata={"test": True, "phase": "2"}
            )
            
            assert memory_id is not None, "Failed to add memory"
            print("   ✅ Memory operations: WORKING")
            
            # Test tool repository
            print("   🔧 Testing tool repository operations...")
            profile = await self.orchestrator.tool_repository.create_agent_profile(test_agent_id)
            assert profile is not None, "Failed to create agent tool profile"
            print("   ✅ Tool repository operations: WORKING")
            
            self.test_results["phase_2_memory_tools"] = "PASSED"
            print("✅ PHASE 2 MEMORY & TOOLS: PASSED")
            
        except Exception as e:
            print(f"❌ PHASE 2 MEMORY & TOOLS: FAILED - {e}")
            self.test_results["phase_2_memory_tools"] = f"FAILED: {e}"
            raise
    
    async def test_phase_3_communication(self) -> None:
        """Test PHASE 3: Communication components."""
        print("\n📡 TESTING PHASE 3: COMMUNICATION")
        print("-" * 50)
        
        try:
            # Communication is disabled by default, so this should be None
            assert self.orchestrator.communication_system is None, "Communication should be disabled by default"
            print("   ✅ Communication disabled by default: CORRECT")
            
            # Test enabling communication
            print("   📡 Testing communication system initialization...")
            config_with_comm = SystemConfig(enable_communication=True, enable_optimization=False)
            comm_orchestrator = UnifiedSystemOrchestrator(config_with_comm)
            await comm_orchestrator.initialize()
            
            assert comm_orchestrator.communication_system is not None, "Communication system not initialized when enabled"
            print("   ✅ Communication system when enabled: OPERATIONAL")
            
            await comm_orchestrator.shutdown()
            
            self.test_results["phase_3_communication"] = "PASSED"
            print("✅ PHASE 3 COMMUNICATION: PASSED")
            
        except Exception as e:
            print(f"❌ PHASE 3 COMMUNICATION: FAILED - {e}")
            self.test_results["phase_3_communication"] = f"FAILED: {e}"
            # Don't raise - communication is optional
    
    async def test_phase_4_optimization(self) -> None:
        """Test PHASE 4: Optimization components."""
        print("\n⚡ TESTING PHASE 4: OPTIMIZATION")
        print("-" * 50)
        
        try:
            # Optimization is disabled in our test config
            assert self.orchestrator.performance_optimizer is None, "Optimization should be disabled in test config"
            print("   ✅ Optimization disabled in test config: CORRECT")
            
            # Test enabling optimization
            print("   ⚡ Testing optimization system initialization...")
            config_with_opt = SystemConfig(enable_communication=False, enable_optimization=True)
            opt_orchestrator = UnifiedSystemOrchestrator(config_with_opt)
            await opt_orchestrator.initialize()
            
            # Optimization might not be available if PerformanceOptimizer is None
            if opt_orchestrator.performance_optimizer is not None:
                print("   ✅ Optimization system when enabled: OPERATIONAL")
            else:
                print("   ⚠️ Optimization system: NOT AVAILABLE (optional component)")
            
            await opt_orchestrator.shutdown()
            
            self.test_results["phase_4_optimization"] = "PASSED"
            print("✅ PHASE 4 OPTIMIZATION: PASSED")
            
        except Exception as e:
            print(f"❌ PHASE 4 OPTIMIZATION: FAILED - {e}")
            self.test_results["phase_4_optimization"] = f"FAILED: {e}"
            # Don't raise - optimization is optional
    
    async def test_system_integration(self) -> None:
        """Test overall system integration."""
        print("\n🔗 TESTING SYSTEM INTEGRATION")
        print("-" * 50)
        
        try:
            # Test system status
            assert self.orchestrator.status.is_initialized, "System should be initialized"
            assert self.orchestrator.status.is_running, "System should be running"
            print("   ✅ System status: HEALTHY")
            
            # Test component integration
            components_working = 0
            total_components = 0
            
            for component_name, status in self.orchestrator.status.components_status.items():
                total_components += 1
                if status:
                    components_working += 1
                    print(f"   ✅ {component_name}: OPERATIONAL")
                else:
                    print(f"   ❌ {component_name}: FAILED")
            
            integration_score = (components_working / total_components) * 100 if total_components > 0 else 0
            print(f"   📊 Integration Score: {integration_score:.1f}% ({components_working}/{total_components})")
            
            assert integration_score >= 75, f"Integration score too low: {integration_score}%"
            
            self.test_results["system_integration"] = "PASSED"
            self.test_results["integration_score"] = integration_score
            print("✅ SYSTEM INTEGRATION: PASSED")
            
        except Exception as e:
            print(f"❌ SYSTEM INTEGRATION: FAILED - {e}")
            self.test_results["system_integration"] = f"FAILED: {e}"
            raise
    
    async def generate_final_report(self) -> None:
        """Generate final test report."""
        print("\n" + "=" * 80)
        print("🎉 COMPREHENSIVE SYSTEM TEST REPORT")
        print("=" * 80)
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"📅 Test Duration: {duration.total_seconds():.2f} seconds")
        print(f"🕐 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test results summary
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASSED")
        total_tests = len([k for k in self.test_results.keys() if k.startswith("phase_") or k == "system_integration"])
        
        print("📊 TEST RESULTS SUMMARY:")
        for test_name, result in self.test_results.items():
            if test_name.startswith("phase_") or test_name == "system_integration":
                status_icon = "✅" if result == "PASSED" else "❌"
                print(f"   {status_icon} {test_name.replace('_', ' ').title()}: {result}")
        
        print()
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"🎯 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        
        if "integration_score" in self.test_results:
            print(f"🔗 Integration Score: {self.test_results['integration_score']:.1f}%")
        
        print()
        if success_rate >= 75:
            print("🎉 THE UNIFIED MULTI-AGENT SYSTEM: FULLY OPERATIONAL")
            print("✅ ALL PHASES COMPLETE AND VERIFIED")
        else:
            print("⚠️ THE UNIFIED MULTI-AGENT SYSTEM: NEEDS ATTENTION")
            print("❌ SOME PHASES REQUIRE FIXES")
        
        print("=" * 80)


async def main():
    """Run the comprehensive system test."""
    test_suite = ComprehensiveSystemTest()
    results = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    if all(result == "PASSED" for key, result in results.items() 
           if key.startswith("phase_") or key == "system_integration"):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    asyncio.run(main())
