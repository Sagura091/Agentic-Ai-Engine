"""
Demonstration of Unlimited Agent Creation and Dynamic Tool Management.

This example showcases the revolutionary capabilities of the enhanced orchestration
system, including unlimited agent creation, dynamic tool generation, and
sophisticated multi-agent workflows.
"""

import asyncio
import json
from typing import Dict, Any, List

from app.orchestration.enhanced_orchestrator import (
    enhanced_orchestrator,
    AgentType,
    OrchestrationStrategy
)
from app.tools.dynamic_tool_factory import ToolCategory, ToolComplexity
from app.tools.production_tool_system import production_tool_registry
from app.core.seamless_integration import seamless_integration


async def demonstrate_unlimited_agents():
    """Demonstrate unlimited agent creation capabilities."""
    print("🚀 Demonstrating Unlimited Agent Creation")
    print("=" * 50)
    
    # Initialize the complete seamless integration system
    await seamless_integration.initialize_complete_system()
    
    # Create various types of agents
    agents_to_create = [
        {
            "type": AgentType.RESEARCH,
            "name": "Research Specialist Alpha",
            "description": "Specialized research agent for scientific literature analysis",
            "config": {"autonomy_level": "autonomous", "learning_mode": "active"}
        },
        {
            "type": AgentType.CREATIVE,
            "name": "Creative Problem Solver Beta",
            "description": "Creative agent for innovative solution generation",
            "config": {"autonomy_level": "emergent", "decision_threshold": 0.4}
        },
        {
            "type": AgentType.OPTIMIZATION,
            "name": "Performance Optimizer Gamma",
            "description": "Optimization agent for system performance enhancement",
            "config": {"autonomy_level": "adaptive", "learning_mode": "reinforcement"}
        },
        {
            "type": AgentType.AUTONOMOUS,
            "name": "General Purpose Agent Delta",
            "description": "Fully autonomous general-purpose agent",
            "config": {"autonomy_level": "autonomous", "enable_goal_setting": True}
        },
        {
            "type": AgentType.BASIC,
            "name": "Task Executor Epsilon",
            "description": "Basic task execution agent",
            "config": {"temperature": 0.3, "max_tokens": 1024}
        }
    ]
    
    created_agents = []
    
    for agent_spec in agents_to_create:
        try:
            agent_id = await enhanced_orchestrator.create_agent_unlimited(
                agent_type=agent_spec["type"],
                name=agent_spec["name"],
                description=agent_spec["description"],
                config=agent_spec["config"]
            )
            created_agents.append(agent_id)
            print(f"✅ Created {agent_spec['type'].value} agent: {agent_spec['name']} (ID: {agent_id})")
        except Exception as e:
            print(f"❌ Failed to create {agent_spec['name']}: {str(e)}")
    
    print(f"\n📊 Total agents created: {len(created_agents)}")
    print(f"📈 System metrics: {enhanced_orchestrator.execution_metrics}")
    
    return created_agents


async def demonstrate_dynamic_tools():
    """Demonstrate dynamic tool creation capabilities."""
    print("\n🛠️ Demonstrating Dynamic Tool Creation")
    print("=" * 50)
    
    # Create tools from templates
    template_tools = [
        ("web_scraper", "Custom Web Scraper", "Scrape content from specific websites"),
        ("api_caller", "REST API Client", "Make HTTP requests to REST APIs"),
        ("json_processor", "JSON Data Processor", "Process and manipulate JSON data")
    ]
    
    created_tools = []
    
    for template_name, custom_name, description in template_tools:
        try:
            tool = await enhanced_orchestrator.tool_factory.create_tool_from_template(
                template_name=template_name,
                custom_name=custom_name,
                custom_description=description
            )
            created_tools.append(tool.name)
            print(f"✅ Created tool from template: {custom_name}")
        except Exception as e:
            print(f"❌ Failed to create tool {custom_name}: {str(e)}")
    
    # Create tools from AI descriptions
    ai_generated_tools = [
        {
            "name": "sentiment_analyzer",
            "description": "Analyze sentiment of text content",
            "functionality": "Take text input and return sentiment analysis with confidence scores, emotion detection, and polarity classification"
        },
        {
            "name": "code_formatter",
            "description": "Format and beautify code",
            "functionality": "Take code in various languages and format it according to best practices, add proper indentation, and improve readability"
        },
        {
            "name": "data_visualizer",
            "description": "Create data visualizations",
            "functionality": "Take structured data and generate various types of charts and graphs, including bar charts, line graphs, and scatter plots"
        }
    ]
    
    for tool_spec in ai_generated_tools:
        try:
            tool = await enhanced_orchestrator.tool_factory.create_tool_from_description(
                name=tool_spec["name"],
                description=tool_spec["description"],
                functionality_description=tool_spec["functionality"],
                llm=enhanced_orchestrator.llm,
                category=ToolCategory.CUSTOM
            )
            created_tools.append(tool.name)
            print(f"✅ Created AI-generated tool: {tool_spec['name']}")
        except Exception as e:
            print(f"❌ Failed to create AI tool {tool_spec['name']}: {str(e)}")
    
    print(f"\n📊 Total tools created: {len(created_tools)}")
    print(f"🔧 Available tool categories: {[cat.value for cat in ToolCategory]}")
    
    return created_tools


async def demonstrate_tool_assignment():
    """Demonstrate dynamic tool assignment to agents."""
    print("\n🔗 Demonstrating Dynamic Tool Assignment")
    print("=" * 50)
    
    # Get available agents and tools
    agents = list(enhanced_orchestrator.agent_registry.keys())
    tools = list(enhanced_orchestrator.tool_factory.registered_tools.keys())
    
    if not agents or not tools:
        print("❌ No agents or tools available for assignment")
        return
    
    # Assign tools to agents strategically
    assignments = [
        (agents[0], ["web_scraper", "api_caller"]),  # Research agent gets web tools
        (agents[1], ["sentiment_analyzer", "data_visualizer"]),  # Creative agent gets analysis tools
        (agents[2], ["code_formatter", "json_processor"]),  # Optimization agent gets processing tools
    ]
    
    for agent_id, tool_names in assignments:
        if agent_id in enhanced_orchestrator.agents:
            try:
                # Filter tools that actually exist
                existing_tools = [tool for tool in tool_names if tool in tools]
                
                if existing_tools:
                    await enhanced_orchestrator.assign_tools_to_agent(agent_id, existing_tools)
                    agent_info = enhanced_orchestrator.agent_registry[agent_id]
                    print(f"✅ Assigned {len(existing_tools)} tools to {agent_info['name']}")
                else:
                    print(f"⚠️ No existing tools to assign to {agent_id}")
            except Exception as e:
                print(f"❌ Failed to assign tools to {agent_id}: {str(e)}")
    
    # Show final tool assignments
    print("\n📋 Final Tool Assignments:")
    for agent_id, agent_tools in enhanced_orchestrator.agent_tools.items():
        if agent_id in enhanced_orchestrator.agent_registry:
            agent_name = enhanced_orchestrator.agent_registry[agent_id]["name"]
            print(f"  {agent_name}: {len(agent_tools)} tools")


async def demonstrate_custom_tool_creation():
    """Demonstrate creating custom tools for specific agents."""
    print("\n🎯 Demonstrating Custom Tool Creation for Agents")
    print("=" * 50)
    
    agents = list(enhanced_orchestrator.agent_registry.keys())
    
    if not agents:
        print("❌ No agents available for custom tool creation")
        return
    
    # Create custom tools for specific agents
    custom_tools = [
        {
            "agent_id": agents[0] if len(agents) > 0 else None,
            "tool_name": "research_paper_analyzer",
            "description": "Analyze research papers for key insights",
            "functionality": "Extract key findings, methodology, conclusions, and citations from research papers in PDF or text format"
        },
        {
            "agent_id": agents[1] if len(agents) > 1 else None,
            "tool_name": "creative_brainstormer",
            "description": "Generate creative ideas and solutions",
            "functionality": "Take a problem description and generate multiple creative solutions using various brainstorming techniques"
        }
    ]
    
    for tool_spec in custom_tools:
        if tool_spec["agent_id"]:
            try:
                tool_name = await enhanced_orchestrator.create_tool_for_agent(
                    agent_id=tool_spec["agent_id"],
                    tool_name=tool_spec["tool_name"],
                    tool_description=tool_spec["description"],
                    functionality_description=tool_spec["functionality"],
                    category=ToolCategory.CUSTOM
                )
                
                agent_name = enhanced_orchestrator.agent_registry[tool_spec["agent_id"]]["name"]
                print(f"✅ Created custom tool '{tool_name}' for {agent_name}")
                
            except Exception as e:
                print(f"❌ Failed to create custom tool: {str(e)}")


async def demonstrate_system_metrics():
    """Demonstrate system metrics and monitoring."""
    print("\n📊 System Metrics and Performance")
    print("=" * 50)
    
    # Get comprehensive metrics
    metrics = {
        "Execution Metrics": enhanced_orchestrator.execution_metrics,
        "Resource Usage": enhanced_orchestrator.resource_usage,
        "Agent Count": len(enhanced_orchestrator.agent_registry),
        "Tool Count": len(enhanced_orchestrator.tool_factory.registered_tools),
        "Global Tool Count": len(enhanced_orchestrator.global_tools),
        "Workflow Templates": list(enhanced_orchestrator.workflow_templates.keys())
    }
    
    print(json.dumps(metrics, indent=2, default=str))
    
    # Show agent performance
    print("\n🎯 Agent Performance Summary:")
    for agent_id, performance in enhanced_orchestrator.agent_performance.items():
        if agent_id in enhanced_orchestrator.agent_registry:
            agent_name = enhanced_orchestrator.agent_registry[agent_id]["name"]
            print(f"  {agent_name}: {performance}")
    
    # Show tool usage statistics
    print("\n🔧 Tool Usage Statistics:")
    for tool_name, tool in enhanced_orchestrator.tool_factory.registered_tools.items():
        print(f"  {tool_name}: {tool.metadata.usage_count} uses, {tool.metadata.success_rate:.2f} success rate")


async def main():
    """Main demonstration function."""
    print("🌟 Revolutionary Agentic AI System Demonstration")
    print("🚀 Unlimited Agents + Dynamic Tools + Enhanced Orchestration")
    print("=" * 70)
    
    try:
        # Run all demonstrations
        await demonstrate_unlimited_agents()
        await demonstrate_dynamic_tools()
        await demonstrate_tool_assignment()
        await demonstrate_custom_tool_creation()
        await demonstrate_system_metrics()
        
        print("\n🎉 Demonstration completed successfully!")
        print("💡 The system now has unlimited agent creation and dynamic tool capabilities!")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
