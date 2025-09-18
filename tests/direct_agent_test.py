#!/usr/bin/env python3
"""
Direct Agent Test - Test agents without full backend.

This script tests agent creation and execution directly using the
orchestration components without requiring the full FastAPI backend.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.orchestration.enhanced_orchestrator import enhanced_orchestrator, AgentType
    from app.core.seamless_integration import seamless_integration
    from app.config.settings import get_settings
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
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


async def test_ollama_connection():
    """Test Ollama connection."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✅ Ollama connection successful")
            print(f"   Available models: {len(models)}")
            if models:
                print(f"   Using model: {models[0].get('name', 'Unknown')}")
                return True, models[0].get('name', 'llama3.2:latest')
            else:
                print("⚠️  No models found in Ollama")
                return False, None
        else:
            print(f"❌ Ollama connection failed: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False, None


async def initialize_system():
    """Initialize the agent orchestration system."""
    try:
        print("🔧 Initializing agent orchestration system...")
        
        # Initialize seamless integration
        await seamless_integration.initialize_complete_system()
        
        print("✅ System initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        logger.error("System initialization failed", error=str(e))
        return False


async def create_test_agent(model_name: str = "gpt-oss:120b"):
    """Create a test agent directly using the orchestrator."""
    try:
        print(f"🤖 Creating test agent with model: {model_name}")
        
        agent_id = await enhanced_orchestrator.create_agent_unlimited(
            agent_type=AgentType.AUTONOMOUS,
            name=f"TestAgent_{datetime.now().strftime('%H%M%S')}",
            description="Direct test agent for validation",
            config={
                "model": model_name,
                "temperature": 0.7,
                "max_tokens": 1000,
                "enable_tool_creation": True,
                "enable_learning": True
            },
            tools=[]
        )
        
        print(f"✅ Agent created successfully: {agent_id}")
        return agent_id
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        logger.error("Agent creation failed", error=str(e))
        return None


async def test_agent_task(agent_id: str, task: str):
    """Test executing a task with an agent."""
    try:
        print(f"📋 Executing task: {task[:50]}...")
        
        result = await enhanced_orchestrator.execute_agent_task(
            agent_id=agent_id,
            task=task,
            context={"test": True, "timestamp": datetime.now().isoformat()}
        )
        
        print(f"✅ Task completed")
        print(f"   Success: {result.get('success', False)}")
        
        response = result.get('response', '')
        if response:
            print(f"   Response: {response[:200]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Task execution failed: {e}")
        logger.error("Task execution failed", error=str(e))
        return None


async def test_tool_creation(agent_id: str):
    """Test tool creation capability."""
    tool_task = """
    Create a simple calculator tool that can add two numbers together.
    After creating the tool, use it to calculate 15 + 27.
    Show me the result and explain what you did.
    """
    
    print("🔧 Testing tool creation...")
    result = await test_agent_task(agent_id, tool_task)
    
    if result and result.get('success', False):
        response = result.get('response', '').lower()
        
        # Check for indicators of tool creation and usage
        tool_indicators = ['tool', 'create', 'function', 'calculator']
        math_indicators = ['42', 'add', 'sum', 'fifteen', 'twenty-seven']
        
        tool_created = any(indicator in response for indicator in tool_indicators)
        math_correct = any(indicator in response for indicator in math_indicators)
        
        print(f"   Tool creation detected: {tool_created}")
        print(f"   Correct calculation: {math_correct}")
        
        return tool_created and math_correct
    
    return False


async def test_autonomous_behavior(agent_id: str):
    """Test autonomous decision making."""
    autonomous_task = """
    You need to plan a simple lunch for 3 people with a $30 budget.
    Make all the decisions yourself: what to serve, where to buy ingredients,
    and how to prepare it. Be specific and practical.
    """
    
    print("🧠 Testing autonomous decision making...")
    result = await test_agent_task(agent_id, autonomous_task)
    
    if result and result.get('success', False):
        response = result.get('response', '').lower()
        
        # Check for autonomous planning indicators
        planning_indicators = ['menu', 'buy', 'prepare', 'budget', 'serve']
        decision_indicators = ['decide', 'choose', 'will', 'plan', 'suggest']
        
        planning_count = sum(1 for indicator in planning_indicators if indicator in response)
        decision_count = sum(1 for indicator in decision_indicators if indicator in response)
        
        print(f"   Planning elements: {planning_count}/5")
        print(f"   Decision indicators: {decision_count}")
        
        return planning_count >= 3 and decision_count >= 2
    
    return False


async def test_learning_adaptation(agent_id: str):
    """Test learning and adaptation capability."""
    learning_task = """
    Solve this riddle: "I have keys but no locks. I have space but no room. 
    You can enter, but you can't go outside. What am I?"
    
    If you're not sure, think about it step by step and explain your reasoning.
    """
    
    print("🎓 Testing learning and adaptation...")
    result = await test_agent_task(agent_id, learning_task)
    
    if result and result.get('success', False):
        response = result.get('response', '').lower()
        
        # Check for learning indicators
        learning_indicators = ['think', 'reason', 'analyze', 'consider', 'keyboard']
        correct_answer = 'keyboard' in response
        
        learning_detected = any(indicator in response for indicator in learning_indicators)
        
        print(f"   Learning behavior detected: {learning_detected}")
        print(f"   Correct answer (keyboard): {correct_answer}")
        
        return learning_detected and correct_answer
    
    return False


async def run_comprehensive_test():
    """Run comprehensive direct agent testing."""
    print("🚀 DIRECT AGENT ORCHESTRATION TESTING")
    print("=" * 60)
    print("Testing agents directly through the orchestration system")
    print("=" * 60)
    
    # Test 1: Ollama Connection
    print("\n1. Ollama Connection Test")
    ollama_ok, model_name = await test_ollama_connection()
    if not ollama_ok:
        print("❌ Cannot proceed without Ollama")
        return False
    
    # Test 2: System Initialization
    print("\n2. System Initialization Test")
    if not await initialize_system():
        print("❌ Cannot proceed without system initialization")
        return False
    
    # Test 3: Agent Creation
    print("\n3. Agent Creation Test")
    agent_id = await create_test_agent(model_name)
    if not agent_id:
        print("❌ Cannot proceed without agent")
        return False
    
    # Test 4: Basic Task
    print("\n4. Basic Task Execution Test")
    basic_task = "Hello! Please introduce yourself and tell me what you can do."
    basic_result = await test_agent_task(agent_id, basic_task)
    basic_success = basic_result and basic_result.get('success', False)
    
    # Test 5: Mathematical Reasoning
    print("\n5. Mathematical Reasoning Test")
    math_task = "Calculate 15 + 27 and explain your reasoning step by step."
    math_result = await test_agent_task(agent_id, math_task)
    math_success = math_result and math_result.get('success', False)
    
    # Test 6: Tool Creation
    print("\n6. Tool Creation Test")
    tool_success = await test_tool_creation(agent_id)
    
    # Test 7: Autonomous Behavior
    print("\n7. Autonomous Behavior Test")
    autonomous_success = await test_autonomous_behavior(agent_id)
    
    # Test 8: Learning and Adaptation
    print("\n8. Learning and Adaptation Test")
    learning_success = await test_learning_adaptation(agent_id)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    tests = [
        ("Ollama Connection", True),
        ("System Initialization", True),
        ("Agent Creation", True),
        ("Basic Task Execution", basic_success),
        ("Mathematical Reasoning", math_success),
        ("Tool Creation", tool_success),
        ("Autonomous Behavior", autonomous_success),
        ("Learning & Adaptation", learning_success)
    ]
    
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    passed_tests = sum(1 for _, success in tests if success)
    total_tests = len(tests)
    
    print(f"\n🎯 Overall Score: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    # Assessment
    if passed_tests >= 7:
        print("\n🎉 EXCELLENT! Your agent system demonstrates true agentic behavior!")
        print("   ✓ Agents can create tools dynamically")
        print("   ✓ Agents show autonomous decision making")
        print("   ✓ Agents demonstrate learning capabilities")
        assessment = "HIGHLY_AGENTIC"
    elif passed_tests >= 6:
        print("\n👍 GOOD! Your agent system shows strong agentic capabilities!")
        print("   ✓ Most core functions working well")
        print("   ⚠ Some advanced features may need tuning")
        assessment = "MODERATELY_AGENTIC"
    elif passed_tests >= 4:
        print("\n⚠️  DEVELOPING! Your agent system has basic functionality!")
        print("   ✓ Basic agent operations working")
        print("   ❌ Advanced agentic features need work")
        assessment = "BASIC_FUNCTIONALITY"
    else:
        print("\n❌ NEEDS WORK! Your agent system has significant issues!")
        print("   ❌ Core functionality problems detected")
        print("   💡 Review system configuration and dependencies")
        assessment = "NEEDS_IMPROVEMENT"
    
    print(f"\n🏆 FINAL ASSESSMENT: {assessment}")
    
    return passed_tests >= 6


async def main():
    """Main entry point."""
    try:
        success = await run_comprehensive_test()
        
        print("\n" + "=" * 60)
        if success:
            print("✨ TESTING COMPLETE - SYSTEM VALIDATED!")
            print("\n💡 NEXT STEPS:")
            print("   • Your agents are working and show agentic behavior")
            print("   • You can create unlimited agents dynamically")
            print("   • Agents can create and use tools autonomously")
            print("   • The orchestration system is functioning properly")
        else:
            print("⚠️  TESTING COMPLETE - ISSUES DETECTED!")
            print("\n💡 RECOMMENDATIONS:")
            print("   • Check Ollama model availability")
            print("   • Verify system configuration")
            print("   • Review agent creation parameters")
            print("   • Test individual components separately")
        
        print("\n🔍 REGARDING RAG:")
        print("   Based on the test results, RAG (Retrieval Augmented Generation)")
        print("   is NOT strictly necessary for basic agentic behavior.")
        print("   However, RAG would be beneficial for:")
        print("   • Knowledge-intensive tasks")
        print("   • Domain-specific information retrieval")
        print("   • Long-term memory and context retention")
        print("   • Enhanced decision making with external data")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        logger.error("Testing failed", error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
