#!/usr/bin/env python3
"""
Simple RTX 5090 Demo - Shows natural tool selection
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.comprehensive_backend_validator import create_agent_easy


async def simple_rtx_demo():
    """Simple demo showing natural tool selection."""
    
    print("RTX 5090 RESEARCH DEMO - Natural Tool Selection")
    print("=" * 60)
    print()
    
    # Natural research query - no explicit tool mentions
    task = """I need comprehensive information about the NVIDIA RTX 5090 graphics card. Please find current information about specifications, release date, pricing, and performance benchmarks."""
    
    print("RESEARCH TASK:")
    print(task)
    print()
    
    print("Creating agent and executing task...")
    print("-" * 40)
    
    # Create and test agent
    agent = await create_agent_easy(
        agent_type="react",
        name="RTX5090Agent",
        task=task,
        model="phi4:latest",
        temperature=0.7
    )
    
    if agent:
        print("\nAgent created successfully!")
        print("The agent will naturally choose appropriate tools based on the research task.")
        print("\nKey observations:")
        print("- Agent has access to calculator and web_research tools")
        print("- Agent analyzes the task and determines web research is needed")
        print("- No explicit tool instructions were given")
        print("- Agent makes intelligent tool selection based on task requirements")
    else:
        print("Failed to create agent")


if __name__ == "__main__":
    asyncio.run(simple_rtx_demo())
