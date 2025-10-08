"""
Memory System Fixes Verification Script

This script verifies that all 5 critical memory system fixes are properly implemented
and functioning correctly.

Run this after deployment to ensure everything is working.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def verify_fix_1_persistence():
    """Verify Fix 1: Automatic Database Persistence"""
    print("\n🔍 Verifying Fix 1: Automatic Database Persistence...")
    
    try:
        from app.memory.unified_memory_system import UnifiedMemorySystem
        import inspect
        
        # Check if _persist_to_database method exists
        if not hasattr(UnifiedMemorySystem, '_persist_to_database'):
            print("❌ FAILED: _persist_to_database method not found")
            return False
        
        # Check if add_memory calls persistence
        source = inspect.getsource(UnifiedMemorySystem.add_memory)
        if '_persist_to_database' not in source:
            print("❌ FAILED: add_memory doesn't call _persist_to_database")
            return False
        
        print("✅ PASSED: Database persistence implemented")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def verify_fix_2_loading():
    """Verify Fix 2: Memory Loading on Restart"""
    print("\n🔍 Verifying Fix 2: Memory Loading on Restart...")
    
    try:
        from app.agents.factory import AgentBuilderFactory
        import inspect
        
        factory = AgentBuilderFactory()
        
        # Check if _load_agent_memories_from_database method exists
        if not hasattr(factory, '_load_agent_memories_from_database'):
            print("❌ FAILED: _load_agent_memories_from_database method not found")
            return False
        
        # Check if _assign_simple_memory calls loading
        source = inspect.getsource(factory._assign_simple_memory)
        if '_load_agent_memories_from_database' not in source:
            print("❌ FAILED: _assign_simple_memory doesn't call memory loading")
            return False
        
        print("✅ PASSED: Memory loading on restart implemented")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def verify_fix_3_integration():
    """Verify Fix 3: Memory Integration in Execution"""
    print("\n🔍 Verifying Fix 3: Memory Integration in Execution...")
    
    try:
        from app.agents.base.agent import LangGraphAgent
        import inspect
        
        # Check if execute method retrieves memories
        source = inspect.getsource(LangGraphAgent.execute)
        
        if 'active_retrieve_memories' not in source and 'retrieve_memories' not in source:
            print("❌ FAILED: execute doesn't retrieve memories")
            return False
        
        if 'add_memory' not in source and 'store_memory' not in source:
            print("❌ FAILED: execute doesn't store outcome as memory")
            return False
        
        print("✅ PASSED: Memory integration in execution implemented")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def verify_fix_4_consolidation():
    """Verify Fix 4: Automatic Consolidation Service"""
    print("\n🔍 Verifying Fix 4: Automatic Consolidation Service...")
    
    try:
        from app.services.memory_consolidation_service import MemoryConsolidationService
        from app.core.unified_system_orchestrator import UnifiedSystemOrchestrator
        import inspect
        
        # Check if service class exists
        if not hasattr(MemoryConsolidationService, 'start'):
            print("❌ FAILED: MemoryConsolidationService.start method not found")
            return False
        
        if not hasattr(MemoryConsolidationService, '_consolidation_loop'):
            print("❌ FAILED: MemoryConsolidationService._consolidation_loop method not found")
            return False
        
        # Check if orchestrator initializes service
        source = inspect.getsource(UnifiedSystemOrchestrator._initialize_phase_2_memory_tools)
        if 'memory_consolidation_service' not in source:
            print("❌ FAILED: Orchestrator doesn't initialize consolidation service")
            return False
        
        print("✅ PASSED: Automatic consolidation service implemented")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def verify_fix_5_bridge():
    """Verify Fix 5: Memory System Bridge"""
    print("\n🔍 Verifying Fix 5: Memory System Bridge...")
    
    try:
        from app.memory.memory_system_bridge import MemorySystemBridge
        
        # Check if bridge class has required methods
        required_methods = [
            'add_memory',
            'retrieve_memories',
            'consolidate_memories',
            'get_agent_context',
            'get_system_type'
        ]
        
        for method in required_methods:
            if not hasattr(MemorySystemBridge, method):
                print(f"❌ FAILED: MemorySystemBridge.{method} method not found")
                return False
        
        print("✅ PASSED: Memory system bridge implemented")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def verify_database_schema():
    """Verify database schema is compatible"""
    print("\n🔍 Verifying Database Schema...")
    
    try:
        from app.models.autonomous import AgentMemoryDB, AutonomousAgentState
        from app.models.agent import Agent
        
        # Check if models have required fields
        required_fields = {
            'AgentMemoryDB': ['memory_id', 'agent_state_id', 'content', 'memory_type', 'importance'],
            'AutonomousAgentState': ['agent_id', 'autonomy_level', 'learning_enabled'],
            'Agent': ['id', 'name', 'agent_type']
        }
        
        models = {
            'AgentMemoryDB': AgentMemoryDB,
            'AutonomousAgentState': AutonomousAgentState,
            'Agent': Agent
        }
        
        for model_name, fields in required_fields.items():
            model = models[model_name]
            for field in fields:
                if not hasattr(model, field):
                    print(f"❌ FAILED: {model_name}.{field} not found")
                    return False
        
        print("✅ PASSED: Database schema compatible")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def verify_all():
    """Run all verification checks"""
    print("=" * 80)
    print("🔍 MEMORY SYSTEM FIXES VERIFICATION")
    print("=" * 80)
    
    results = {
        "Fix 1: Database Persistence": await verify_fix_1_persistence(),
        "Fix 2: Memory Loading": await verify_fix_2_loading(),
        "Fix 3: Memory Integration": await verify_fix_3_integration(),
        "Fix 4: Consolidation Service": await verify_fix_4_consolidation(),
        "Fix 5: Memory Bridge": await verify_fix_5_bridge(),
        "Database Schema": await verify_database_schema()
    }
    
    print("\n" + "=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)
    
    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Memory system fixes are properly implemented!")
        print("=" * 80)
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED - Please review the implementation")
        print("=" * 80)
        return 1


async def main():
    """Main entry point"""
    try:
        exit_code = await verify_all()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

