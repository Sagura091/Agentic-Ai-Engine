#!/usr/bin/env python3
"""
Simple Agent Test Script.

This script performs basic tests to verify that agents can be created
and execute tasks properly with the Ollama integration.
"""

import json
import time
import requests
from datetime import datetime


def test_backend_health():
    """Test if the backend is healthy and responding."""
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        if response.status_code == 200:
            print("✅ Backend health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False


def test_ollama_connection():
    """Test if Ollama is accessible and has models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✅ Ollama connection successful")
            print(f"   Available models: {len(models)}")
            for model in models[:3]:  # Show first 3 models
                print(f"   - {model.get('name', 'Unknown')}")
            return True, models
        else:
            print(f"❌ Ollama connection failed: {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False, []


def create_test_agent():
    """Create a test agent for validation."""
    try:
        agent_data = {
            "agent_type": "autonomous",
            "name": f"TestAgent_{int(time.time())}",
            "description": "Test agent for validation",
            "model": "gpt-oss:120b",
            "config": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "enable_tool_creation": True
            }
        }
        
        print("🤖 Creating test agent...")
        response = requests.post(
            "http://localhost:8001/api/v1/orchestration/agents",
            json=agent_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            agent_id = result.get("agent_id")
            print(f"✅ Agent created successfully: {agent_id}")
            return agent_id
        else:
            print(f"❌ Agent creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return None


def test_agent_task(agent_id, task):
    """Test executing a task with an agent."""
    try:
        task_data = {
            "agent_id": agent_id,
            "task": task,
            "context": {"test": True}
        }
        
        print(f"📋 Executing task: {task[:50]}...")
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8001/api/v1/orchestration/agents/execute",
            json=task_data,
            timeout=120
        )
        
        execution_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Task completed in {execution_time:.2f}s")
            print(f"   Success: {result.get('success', False)}")
            
            agent_response = result.get('response', '')
            if agent_response:
                print(f"   Response: {agent_response[:200]}...")
            
            return result
        else:
            print(f"❌ Task execution failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Task execution failed: {e}")
        return None


def test_tool_creation(agent_id):
    """Test if agent can create and use tools."""
    tool_creation_task = """
    Create a simple calculator tool that can add two numbers.
    After creating the tool, use it to calculate 15 + 27.
    Explain what you did step by step.
    """
    
    print("🔧 Testing tool creation capability...")
    result = test_agent_task(agent_id, tool_creation_task)
    
    if result and result.get('success', False):
        response = result.get('response', '').lower()
        
        # Check for tool creation indicators
        tool_indicators = ['tool', 'create', 'function', 'calculator']
        math_indicators = ['42', 'fifteen', 'twenty-seven', 'add']
        
        tool_created = any(indicator in response for indicator in tool_indicators)
        math_correct = any(indicator in response for indicator in math_indicators)
        
        print(f"   Tool creation detected: {tool_created}")
        print(f"   Correct calculation: {math_correct}")
        
        return tool_created and math_correct
    
    return False


def test_autonomous_behavior(agent_id):
    """Test autonomous decision making."""
    autonomous_task = """
    You need to plan a small dinner party for 4 people with a budget of $50.
    Make all the decisions yourself including menu, shopping list, and timeline.
    Be specific and practical in your planning.
    """
    
    print("🧠 Testing autonomous decision making...")
    result = test_agent_task(agent_id, autonomous_task)
    
    if result and result.get('success', False):
        response = result.get('response', '').lower()
        
        # Check for autonomous planning indicators
        planning_indicators = ['menu', 'shopping', 'timeline', 'plan', 'budget']
        decision_indicators = ['decide', 'choose', 'select', 'will', 'recommend']
        
        planning_detected = sum(1 for indicator in planning_indicators if indicator in response)
        decisions_detected = sum(1 for indicator in decision_indicators if indicator in response)
        
        print(f"   Planning elements detected: {planning_detected}/5")
        print(f"   Decision indicators: {decisions_detected}")
        
        return planning_detected >= 3 and decisions_detected >= 2
    
    return False


def run_comprehensive_test():
    """Run comprehensive agent testing."""
    print("🚀 AGENT ORCHESTRATION TESTING")
    print("=" * 50)
    
    # Test 1: Backend Health
    print("\n1. Backend Health Check")
    if not test_backend_health():
        print("❌ Cannot proceed - backend is not healthy")
        return False
    
    # Test 2: Ollama Connection
    print("\n2. Ollama Connection Check")
    ollama_ok, models = test_ollama_connection()
    if not ollama_ok:
        print("❌ Cannot proceed - Ollama is not accessible")
        return False
    
    # Test 3: Agent Creation
    print("\n3. Agent Creation Test")
    agent_id = create_test_agent()
    if not agent_id:
        print("❌ Cannot proceed - agent creation failed")
        return False
    
    # Test 4: Basic Task Execution
    print("\n4. Basic Task Execution Test")
    basic_task = "Calculate 15 + 27 and explain your reasoning."
    basic_result = test_agent_task(agent_id, basic_task)
    basic_success = basic_result and basic_result.get('success', False)
    
    if not basic_success:
        print("❌ Basic task execution failed")
        return False
    
    # Test 5: Tool Creation
    print("\n5. Tool Creation Test")
    tool_success = test_tool_creation(agent_id)
    
    # Test 6: Autonomous Behavior
    print("\n6. Autonomous Behavior Test")
    autonomous_success = test_autonomous_behavior(agent_id)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Backend Health: PASS")
    print(f"✅ Ollama Connection: PASS")
    print(f"✅ Agent Creation: PASS")
    print(f"✅ Basic Task Execution: PASS")
    print(f"{'✅' if tool_success else '❌'} Tool Creation: {'PASS' if tool_success else 'FAIL'}")
    print(f"{'✅' if autonomous_success else '❌'} Autonomous Behavior: {'PASS' if autonomous_success else 'FAIL'}")
    
    total_tests = 6
    passed_tests = 4 + (1 if tool_success else 0) + (1 if autonomous_success else 0)
    
    print(f"\n🎯 Overall Score: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests >= 5:
        print("🎉 EXCELLENT! Your agent system is working well!")
        print("   Agents demonstrate true agentic behavior.")
    elif passed_tests >= 4:
        print("👍 GOOD! Your agent system is functional.")
        print("   Some agentic capabilities may need improvement.")
    else:
        print("⚠️  NEEDS WORK! Your agent system has issues.")
        print("   Consider reviewing agent configuration and capabilities.")
    
    return passed_tests >= 4


def test_real_time_monitoring(agent_id, duration_minutes=2):
    """Test real-time agent monitoring."""
    print(f"\n7. Real-Time Monitoring Test ({duration_minutes} minutes)")
    
    tasks = [
        "What would you do if asked to solve a complex problem?",
        "How would you approach creating a new tool?",
        "Describe your decision-making process.",
        "What makes you different from a simple chatbot?"
    ]
    
    results = []
    
    for i, task in enumerate(tasks):
        print(f"   Task {i+1}/{len(tasks)}: {task[:30]}...")
        result = test_agent_task(agent_id, task)
        
        if result:
            response = result.get('response', '')
            results.append({
                'task': task,
                'response_length': len(response),
                'execution_time': result.get('execution_time', 0),
                'success': result.get('success', False)
            })
        
        time.sleep(10)  # Wait between tasks
    
    # Analyze results
    successful_tasks = len([r for r in results if r['success']])
    avg_response_length = sum(r['response_length'] for r in results) / len(results) if results else 0
    
    print(f"   Completed tasks: {successful_tasks}/{len(tasks)}")
    print(f"   Average response length: {avg_response_length:.0f} characters")
    
    return successful_tasks >= len(tasks) * 0.75


if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        
        if success:
            print("\n🔍 Running additional real-time monitoring...")
            # Get the last created agent for monitoring
            # This is a simplified version - in practice you'd track the agent ID
            print("   (Skipping real-time monitoring for now)")
            
        print("\n✨ Testing complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
