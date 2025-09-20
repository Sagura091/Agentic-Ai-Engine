#!/usr/bin/env python3
"""
Show Full RTX 5090 Research Conversation
Demonstrates natural tool selection and complete agent thinking process.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.comprehensive_backend_validator import ComprehensiveBackendValidator
from app.agents.factory import AgentType


async def show_full_rtx5090_conversation():
    """Show the complete conversation with detailed agent thinking."""
    
    print("=" * 80)
    print("RTX 5090 RESEARCH - FULL CONVERSATION WITH NATURAL TOOL SELECTION")
    print("=" * 80)
    print()
    
    # Initialize validator
    print("Initializing system...")
    validator = ComprehensiveBackendValidator()
    await validator.initialize()
    print("System ready!")
    print()
    
    # Create agent with natural tool selection capabilities
    print("Creating RTX 5090 Research Agent...")
    agent = await validator.create_agent_on_demand(
        agent_type=AgentType.REACT,
        custom_name="RTX5090ResearchAgent",
        model_name="phi4:latest",
        temperature=0.7,
        custom_task="I need comprehensive information about the NVIDIA RTX 5090 graphics card including specifications, release date, pricing, and performance benchmarks"
    )
    
    if not agent:
        print("Failed to create agent!")
        return
    
    print(f"Agent created: {agent.agent_id}")
    print(f"Available tools: {[tool.name for tool in agent.tools]}")
    print()
    
    # Natural research query - no explicit tool mentions
    task = """I need comprehensive information about the NVIDIA RTX 5090 graphics card. Please find:

1. Official specifications and technical details
2. Expected release date and availability
3. Pricing information and market positioning
4. Performance benchmarks and comparisons with RTX 4090
5. Key features and architectural improvements

Please research this thoroughly and provide detailed analysis with current information."""
    
    print("RESEARCH TASK:")
    print("-" * 40)
    print(task)
    print()
    
    print("AGENT THINKING AND RESPONSE:")
    print("=" * 60)
    
    # Execute task and capture full response
    try:
        result = await agent.execute_task(task)
        
        if result.success:
            print("FULL AGENT RESPONSE:")
            print("-" * 40)
            print(result.response)
            print()
            
            print("EXECUTION DETAILS:")
            print(f"- Success: {result.success}")
            print(f"- Iterations: {result.iterations}")
            print(f"- Messages: {len(result.messages)}")
            print(f"- Execution time: {result.execution_time:.2f}s")
            
            if result.tool_calls:
                print(f"- Tools used: {len(result.tool_calls)}")
                for i, tool_call in enumerate(result.tool_calls, 1):
                    print(f"  {i}. {tool_call}")
            else:
                print("- No tools were used")
                
            # Show the conversation flow
            print("\nCONVERSATION FLOW:")
            print("-" * 30)
            for i, message in enumerate(result.messages, 1):
                if hasattr(message, 'content'):
                    content = message.content[:200] + "..." if len(message.content) > 200 else message.content
                    print(f"{i}. {type(message).__name__}: {content}")
                    
        else:
            print(f"Task failed: {result.error}")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("CONVERSATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(show_full_rtx5090_conversation())
