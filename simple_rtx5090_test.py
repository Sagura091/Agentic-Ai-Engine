#!/usr/bin/env python3
"""
Simple RTX 5090 Research Test - Shows natural tool selection
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator, get_orchestrator_with_compatibility


async def test_natural_tool_selection():
    """Test natural tool selection without explicit tool mentions."""
    
    print("RTX 5090 RESEARCH TEST - Natural Tool Selection")
    print("=" * 50)
    print()
    
    # Initialize system
    print("Initializing system...")
    orchestrator = get_orchestrator_with_compatibility()
    await orchestrator.initialize()
    print("System initialized!")
    print()

    # Create agent using compatibility layer
    print("Creating research agent...")
    agent_id = await orchestrator.create_agent(
        agent_type="react",
        config={
            "name": "RTX5090ResearchAgent",
            "description": "Research agent for graphics card information",
            "model": "phi4:latest",
            "temperature": 0.7,
            "max_tokens": 2000,
            "tools": ["calculator", "web_research"]
        }
    )
    
    print(f"Agent created: {agent_id}")
    print()
    
    # Natural research task - no explicit tool mentions
    task = """I need comprehensive information about the NVIDIA RTX 5090 graphics card. Please find:

1. Official specifications and technical details
2. Expected release date and availability
3. Pricing information and market positioning  
4. Performance benchmarks and comparisons
5. Key features and improvements over RTX 4090

Please research this thoroughly and provide detailed analysis with current information."""
    
    print("RESEARCH TASK:")
    print(task)
    print()
    print("AGENT RESPONSE:")
    print("-" * 30)
    
    # Execute task
    try:
        result = await orchestrator.execute_agent_task(
            agent_id=agent_id,
            task=task,
            context={"test_mode": True}
        )
        
        if result.get("success"):
            response = result.get("response", "")
            print(response)
            print()
            
            # Show tool usage if any
            if "tool_calls" in result:
                print("TOOLS USED:")
                for tool_call in result["tool_calls"]:
                    print(f"- {tool_call}")
            
        else:
            print(f"Task failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nTest complete!")


if __name__ == "__main__":
    asyncio.run(test_natural_tool_selection())
