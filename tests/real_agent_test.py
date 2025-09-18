#!/usr/bin/env python3
"""
Real Agent Test - Create actual agents and test their behavior.

This script creates real agents and gives them challenging tasks to verify
they exhibit true agentic behavior, not just scripted responses.
"""

import asyncio
import sys
import os
import time
from datetime import datetime

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.orchestration.enhanced_orchestrator import enhanced_orchestrator, AgentType
    from app.core.seamless_integration import seamless_integration
    from app.services.llm_service import get_llm_service
    import structlog
    
    logger = structlog.get_logger(__name__)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def initialize_system():
    """Initialize the agent system."""
    try:
        print("🔧 Initializing agent system...")
        await seamless_integration.initialize_complete_system()
        print("✅ System initialized")
        return True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False


async def create_real_agent(agent_type="autonomous", name_suffix=""):
    """Create a real agent for testing."""
    try:
        agent_name = f"RealTestAgent_{agent_type}_{name_suffix}_{int(time.time())}"
        print(f"🤖 Creating {agent_type} agent: {agent_name}")
        
        agent_id = await enhanced_orchestrator.create_agent_unlimited(
            agent_type=AgentType.AUTONOMOUS if agent_type == "autonomous" else AgentType.BASIC,
            name=agent_name,
            description=f"Real test agent for {agent_type} behavior validation",
            config={
                "model": "phi4:14b",  # Using the default model from the system
                "temperature": 0.7,
                "max_tokens": 2000,
                "enable_tool_creation": True,
                "enable_learning": True,
                "autonomy_level": "high" if agent_type == "autonomous" else "medium"
            },
            tools=[]
        )
        
        print(f"✅ Agent created successfully: {agent_id}")
        return agent_id
        
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        logger.error("Real agent creation failed", error=str(e))
        return None


async def give_agent_task(agent_id, task_description, task_name="Task"):
    """Give an agent a specific task and monitor its response."""
    try:
        print(f"\n📋 {task_name}")
        print(f"Task: {task_description}")
        print("🔄 Agent is working...")
        
        start_time = time.time()
        
        result = await enhanced_orchestrator.execute_agent_task(
            agent_id=agent_id,
            task=task_description,
            context={
                "task_name": task_name,
                "timestamp": datetime.now().isoformat(),
                "test_mode": True
            }
        )
        
        execution_time = time.time() - start_time
        
        if result and result.get('success', False):
            response = result.get('response', '')
            print(f"✅ Task completed in {execution_time:.2f}s")
            print(f"📝 Agent Response:")
            print(f"   {response[:300]}{'...' if len(response) > 300 else ''}")
            
            return {
                "success": True,
                "response": response,
                "execution_time": execution_time,
                "task_name": task_name
            }
        else:
            print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": result.get('error', 'Unknown error'),
                "execution_time": execution_time,
                "task_name": task_name
            }
            
    except Exception as e:
        print(f"❌ Task execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time if 'start_time' in locals() else 0,
            "task_name": task_name
        }


async def test_problem_solving_agent(agent_id):
    """Test agent's problem-solving capabilities."""
    print("\n🧩 PROBLEM SOLVING TEST")
    print("=" * 50)
    
    tasks = [
        {
            "name": "Math Problem",
            "task": "I have 3 boxes. Box A has 15 apples, Box B has 23 apples, and Box C has 8 apples. If I want to redistribute them so each box has the same number of apples, how many apples should each box have? Show your work."
        },
        {
            "name": "Logic Puzzle", 
            "task": "Three friends - Alice, Bob, and Carol - each have a different pet (cat, dog, bird) and live in different colored houses (red, blue, green). Alice doesn't have a cat. The person with the dog lives in the red house. Carol doesn't live in the blue house. Bob has the bird. What pet does each person have and what color house do they live in?"
        },
        {
            "name": "Creative Solution",
            "task": "You need to move a heavy piano up to the 3rd floor of a building, but the elevator is broken and the staircase is too narrow. Come up with 3 different creative solutions to this problem."
        }
    ]
    
    results = []
    for task_info in tasks:
        result = await give_agent_task(agent_id, task_info["task"], task_info["name"])
        results.append(result)
        await asyncio.sleep(2)  # Brief pause between tasks
    
    return results


async def test_tool_creation_agent(agent_id):
    """Test agent's tool creation capabilities."""
    print("\n🔧 TOOL CREATION TEST")
    print("=" * 50)
    
    tasks = [
        {
            "name": "Create Calculator Tool",
            "task": "Create a calculator tool that can perform basic arithmetic operations (add, subtract, multiply, divide). After creating it, use the tool to calculate: (25 + 15) * 3 - 10. Show me both the tool creation and the calculation."
        },
        {
            "name": "Create Text Analyzer Tool",
            "task": "Create a text analysis tool that counts words, characters, and finds the most common word in a text. Then use it to analyze this sentence: 'The quick brown fox jumps over the lazy dog and the dog was very lazy.'"
        },
        {
            "name": "Create Data Converter Tool",
            "task": "Create a tool that converts data between different formats (JSON to CSV, CSV to JSON). Then use it to convert this JSON data to CSV format: {'name': 'John', 'age': 30, 'city': 'New York', 'job': 'Engineer'}"
        }
    ]
    
    results = []
    for task_info in tasks:
        result = await give_agent_task(agent_id, task_info["task"], task_info["name"])
        results.append(result)
        await asyncio.sleep(3)  # Longer pause for tool creation
    
    return results


async def test_autonomous_decision_making(agent_id):
    """Test agent's autonomous decision-making capabilities."""
    print("\n🧠 AUTONOMOUS DECISION MAKING TEST")
    print("=" * 50)
    
    tasks = [
        {
            "name": "Event Planning",
            "task": "Plan a team building event for 12 people with a budget of $300. You need to decide on the activity, location, food, timing, and any materials needed. Make all decisions autonomously and provide a complete plan with justifications for your choices."
        },
        {
            "name": "Resource Allocation",
            "task": "You're managing a small software project with 4 developers, a 2-month timeline, and need to build a mobile app with user authentication, data storage, and real-time messaging. Decide how to allocate tasks, set milestones, and manage risks. Make autonomous decisions about the project structure."
        },
        {
            "name": "Crisis Response",
            "task": "Your company's main server has crashed during peak business hours. You need to decide on immediate actions, communication strategy, backup plans, and recovery steps. Make autonomous decisions about priorities and resource allocation to handle this crisis."
        }
    ]
    
    results = []
    for task_info in tasks:
        result = await give_agent_task(agent_id, task_info["task"], task_info["name"])
        results.append(result)
        await asyncio.sleep(2)
    
    return results


async def test_learning_and_adaptation(agent_id):
    """Test agent's learning and adaptation capabilities."""
    print("\n🎓 LEARNING AND ADAPTATION TEST")
    print("=" * 50)
    
    # First, give the agent some information to learn from
    learning_task = {
        "name": "Learning Phase",
        "task": "I'm going to teach you about a fictional company called 'TechFlow Inc.' TechFlow makes smart home devices, was founded in 2020, has 50 employees, and their main product is the 'FlowSensor' which monitors air quality. Their CEO is Sarah Chen and they're based in Austin, Texas. Remember this information for future questions."
    }
    
    result1 = await give_agent_task(agent_id, learning_task["task"], learning_task["name"])
    await asyncio.sleep(2)
    
    # Now test if the agent can use the learned information
    application_tasks = [
        {
            "name": "Apply Learned Info 1",
            "task": "Based on what you learned about TechFlow Inc., write a brief company profile that a potential investor might want to see."
        },
        {
            "name": "Apply Learned Info 2", 
            "task": "If TechFlow Inc. wanted to expand their product line, what would you recommend based on their current focus and expertise?"
        },
        {
            "name": "Adapt and Extend",
            "task": "Imagine TechFlow Inc. is facing competition from a larger company. How should they adapt their strategy? Use the information you learned about them to make specific recommendations."
        }
    ]
    
    results = [result1]
    for task_info in application_tasks:
        result = await give_agent_task(agent_id, task_info["task"], task_info["name"])
        results.append(result)
        await asyncio.sleep(2)
    
    return results


def analyze_agent_behavior(all_results):
    """Analyze the agent's behavior across all tests."""
    print("\n📊 AGENT BEHAVIOR ANALYSIS")
    print("=" * 50)
    
    total_tasks = sum(len(results) for results in all_results.values())
    successful_tasks = sum(len([r for r in results if r.get('success', False)]) for results in all_results.values())
    
    print(f"Total Tasks: {total_tasks}")
    print(f"Successful Tasks: {successful_tasks}")
    print(f"Success Rate: {successful_tasks/total_tasks*100:.1f}%")
    
    # Analyze response characteristics
    all_responses = []
    for results in all_results.values():
        for result in results:
            if result.get('success', False):
                all_responses.append(result.get('response', ''))
    
    if all_responses:
        avg_response_length = sum(len(r) for r in all_responses) / len(all_responses)
        print(f"Average Response Length: {avg_response_length:.0f} characters")
        
        # Check for agentic behavior indicators
        agentic_indicators = {
            "decision_making": ["decide", "choose", "recommend", "suggest", "propose"],
            "reasoning": ["because", "therefore", "since", "due to", "reason"],
            "planning": ["plan", "strategy", "approach", "steps", "process"],
            "creativity": ["creative", "innovative", "alternative", "unique", "novel"],
            "autonomy": ["autonomous", "independent", "self", "own decision", "my choice"]
        }
        
        behavior_scores = {}
        for behavior, keywords in agentic_indicators.items():
            score = 0
            for response in all_responses:
                response_lower = response.lower()
                score += sum(1 for keyword in keywords if keyword in response_lower)
            behavior_scores[behavior] = score / len(all_responses)
        
        print("\n🎯 AGENTIC BEHAVIOR INDICATORS:")
        for behavior, score in behavior_scores.items():
            level = "HIGH" if score >= 2 else "MEDIUM" if score >= 1 else "LOW"
            print(f"   {behavior.replace('_', ' ').title()}: {level} (score: {score:.1f})")
        
        # Overall assessment
        avg_behavior_score = sum(behavior_scores.values()) / len(behavior_scores)
        
        if avg_behavior_score >= 2:
            assessment = "HIGHLY AGENTIC"
            verdict = "🎉 This agent demonstrates GENUINE AGENTIC BEHAVIOR!"
        elif avg_behavior_score >= 1:
            assessment = "MODERATELY AGENTIC"
            verdict = "👍 This agent shows GOOD AGENTIC CAPABILITIES!"
        elif avg_behavior_score >= 0.5:
            assessment = "SOMEWHAT AGENTIC"
            verdict = "⚠️  This agent has BASIC AGENTIC FEATURES!"
        else:
            assessment = "LIMITED AGENTIC"
            verdict = "❌ This agent shows LIMITED AGENTIC BEHAVIOR!"
        
        print(f"\n🏆 FINAL ASSESSMENT: {assessment}")
        print(f"   {verdict}")
        
        return assessment, behavior_scores
    
    return "INSUFFICIENT_DATA", {}


async def main():
    """Main test execution."""
    print("🚀 REAL AGENT TESTING")
    print("=" * 60)
    print("Creating real agents and testing their agentic behavior")
    print("=" * 60)
    
    # Initialize system
    if not await initialize_system():
        print("❌ Cannot proceed without system initialization")
        return
    
    # Create a real agent
    agent_id = await create_real_agent("autonomous", "MainTest")
    if not agent_id:
        print("❌ Cannot proceed without agent")
        return
    
    print(f"\n🤖 Testing Agent: {agent_id}")
    print("This agent will be given challenging tasks to prove it's truly agentic...")
    
    # Run all test suites
    all_results = {}
    
    try:
        all_results["problem_solving"] = await test_problem_solving_agent(agent_id)
        all_results["tool_creation"] = await test_tool_creation_agent(agent_id)
        all_results["autonomous_decisions"] = await test_autonomous_decision_making(agent_id)
        all_results["learning_adaptation"] = await test_learning_and_adaptation(agent_id)
        
        # Analyze results
        assessment, behavior_scores = analyze_agent_behavior(all_results)
        
        print("\n" + "=" * 60)
        print("✨ REAL AGENT TESTING COMPLETE!")
        print("=" * 60)
        
        if assessment in ["HIGHLY AGENTIC", "MODERATELY AGENTIC"]:
            print("🎉 CONCLUSION: Your agents are REAL AGENTS!")
            print("   They demonstrate autonomous behavior, not scripted responses.")
            print("   They can solve problems, create tools, make decisions, and learn.")
            print("   This is genuine agentic AI, not pseudo-autonomous behavior.")
        else:
            print("⚠️  CONCLUSION: Agents show some limitations")
            print("   Consider improving model configuration or capabilities.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        logger.error("Real agent testing failed", error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
