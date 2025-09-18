#!/usr/bin/env python3
"""
Agentic AI vs Regular LLM Comparison Test
This test demonstrates the clear difference between regular LLM responses and true agentic behavior.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Import real backend tools
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools import calculator_tool, business_intelligence_tool
from app.agents.base.agent import LangGraphAgent, AgentConfig
from app.agents.autonomous.autonomous_agent import AutonomousLangGraphAgent, AutonomousAgentConfig
from app.agents.autonomous.autonomous_agent import AutonomyLevel, LearningMode
from app.agents.base.agent import AgentCapability

class AgentBehaviorComparison:
    """Compare regular LLM responses vs agentic behavior."""
    
    def __init__(self):
        self.results = {
            "regular_llm": [],
            "basic_agent": [],
            "agentic_agent": []
        }
        self.start_time = datetime.now()
    
    async def create_llm(self):
        """Create the LLM for testing."""
        # Import the OllamaHTTPLLM from the comprehensive test
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        try:
            from comprehensive_agent_behavior_analysis import OllamaHTTPLLM, get_available_ollama_models
        except ImportError:
            # Fallback: create a simple Ollama LLM implementation
            from app.http_client import SimpleHTTPClient
            from langchain_core.language_models.base import BaseLanguageModel
            from langchain_core.messages import BaseMessage, AIMessage
            from typing import List, Any, Optional
            import json

            class OllamaHTTPLLM(BaseLanguageModel):
                def __init__(self, model: str = "phi4:14b"):
                    super().__init__()
                    self.model = model
                    self.client = SimpleHTTPClient()

                async def _agenerate(self, messages: List[BaseMessage], **kwargs) -> Any:
                    # Convert messages to prompt
                    prompt = messages[-1].content if messages else ""

                    # Make request to Ollama
                    response = await self.client.post_json(
                        "http://localhost:11434/api/generate",
                        {
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False
                        }
                    )

                    return AIMessage(content=response.get("response", ""))

                async def ainvoke(self, prompt: str, **kwargs) -> AIMessage:
                    response = await self.client.post_json(
                        "http://localhost:11434/api/generate",
                        {
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False
                        }
                    )

                    return AIMessage(content=response.get("response", ""))

                def _generate(self, messages: List[BaseMessage], **kwargs) -> Any:
                    import asyncio
                    return asyncio.run(self._agenerate(messages, **kwargs))

                @property
                def _llm_type(self) -> str:
                    return "ollama_http"

            async def get_available_ollama_models():
                client = SimpleHTTPClient()
                try:
                    response = await client.get_json("http://localhost:11434/api/tags")
                    return [model["name"] for model in response.get("models", [])]
                except:
                    return ["phi4:14b"]  # Fallback

        models = await get_available_ollama_models()
        if not models:
            raise Exception("No Ollama models available")

        # Prefer phi4:14b for testing
        selected_model = "phi4:14b" if "phi4:14b" in models else models[0]
        print(f"🤖 Using Ollama model: {selected_model}")

        return OllamaHTTPLLM(model=selected_model)
    
    async def test_regular_llm_response(self, prompt: str) -> Dict[str, Any]:
        """Test regular LLM response without agent infrastructure."""
        print(f"\n🔤 REGULAR LLM TEST")
        print(f"Prompt: {prompt[:100]}...")
        
        start_time = time.time()
        llm = await self.create_llm()
        
        try:
            response = await llm.ainvoke(prompt)
            execution_time = time.time() - start_time
            
            result = {
                "type": "regular_llm",
                "prompt": prompt,
                "response": response.content,
                "execution_time": execution_time,
                "tools_used": [],
                "reasoning_steps": 0,
                "autonomous_actions": 0,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ LLM Response: {response.content[:150]}...")
            print(f"⏱️  Execution time: {execution_time:.2f}s")
            
            self.results["regular_llm"].append(result)
            return result
            
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return {"error": str(e)}
    
    async def test_basic_agent_with_tools(self, prompt: str) -> Dict[str, Any]:
        """Test basic agent with access to tools."""
        print(f"\n🤖 BASIC AGENT TEST")
        print(f"Prompt: {prompt[:100]}...")
        
        start_time = time.time()
        
        try:
            # Create basic agent with tools
            config = AgentConfig(
                name="Tool-Using Agent",
                description="Agent that can use tools to solve problems",
                model_name="phi4:14b",
                model_provider="ollama",
                capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE]
            )
            
            llm = await self.create_llm()
            
            # Use real backend tools as a list
            tools = [calculator_tool, business_intelligence_tool]
            
            agent = LangGraphAgent(config=config, llm=llm, tools=tools)
            
            # Execute with tool access
            result = await agent.execute(prompt)
            execution_time = time.time() - start_time
            
            # Analyze tool usage
            tools_used = result.get("tool_calls", [])
            reasoning_steps = result.get("iteration_count", 0)
            
            agent_result = {
                "type": "basic_agent",
                "prompt": prompt,
                "response": result,
                "execution_time": execution_time,
                "tools_used": [tool.get("tool_name", "unknown") for tool in tools_used],
                "reasoning_steps": reasoning_steps,
                "autonomous_actions": len(tools_used),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Agent Response: {len(result.get('messages', []))} messages")
            print(f"🔧 Tools Used: {len(tools_used)}")
            print(f"🧠 Reasoning Steps: {reasoning_steps}")
            print(f"⏱️  Execution time: {execution_time:.2f}s")
            
            self.results["basic_agent"].append(agent_result)
            return agent_result
            
        except Exception as e:
            print(f"❌ Agent Error: {e}")
            return {"error": str(e)}
    
    async def test_autonomous_agent(self, prompt: str) -> Dict[str, Any]:
        """Test autonomous agent with full agentic capabilities."""
        print(f"\n🚀 AUTONOMOUS AGENT TEST")
        print(f"Prompt: {prompt[:100]}...")
        
        start_time = time.time()
        
        try:
            # Create autonomous agent
            config = AutonomousAgentConfig(
                name="Autonomous Problem Solver",
                description="Fully autonomous agent with learning and adaptation",
                model_name="phi4:14b",
                model_provider="ollama",
                capabilities=[
                    AgentCapability.REASONING, 
                    AgentCapability.TOOL_USE, 
                    AgentCapability.PLANNING,
                    AgentCapability.LEARNING
                ],
                autonomy_level=AutonomyLevel.AUTONOMOUS,
                learning_mode=LearningMode.ACTIVE,
                enable_proactive_behavior=True
            )
            
            llm = await self.create_llm()
            
            # Use real backend tools as a list
            tools = [calculator_tool, business_intelligence_tool]
            
            agent = AutonomousLangGraphAgent(config=config, llm=llm, tools=tools)
            
            # Execute with full autonomy
            result = await agent.execute(prompt)
            execution_time = time.time() - start_time
            
            # Analyze autonomous behavior
            tools_used = result.get("tool_calls", [])
            reasoning_steps = result.get("iteration_count", 0)
            autonomous_actions = len(tools_used) + len(result.get("goal_stack", []))
            
            autonomous_result = {
                "type": "autonomous_agent",
                "prompt": prompt,
                "response": result,
                "execution_time": execution_time,
                "tools_used": [tool.get("tool_name", "unknown") for tool in tools_used],
                "reasoning_steps": reasoning_steps,
                "autonomous_actions": autonomous_actions,
                "goals_created": len(result.get("goal_stack", [])),
                "learning_enabled": result.get("learning_enabled", False),
                "adaptation_history": len(result.get("adaptation_history", [])),
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"✅ Autonomous Response: {len(result.get('messages', []))} messages")
            print(f"🔧 Tools Used: {len(tools_used)}")
            print(f"🎯 Goals Created: {len(result.get('goal_stack', []))}")
            print(f"🧠 Reasoning Steps: {reasoning_steps}")
            print(f"🔄 Autonomous Actions: {autonomous_actions}")
            print(f"⏱️  Execution time: {execution_time:.2f}s")
            
            self.results["agentic_agent"].append(autonomous_result)
            return autonomous_result
            
        except Exception as e:
            print(f"❌ Autonomous Agent Error: {e}")
            return {"error": str(e)}
    
    async def run_comparison_test(self):
        """Run comprehensive comparison test."""
        print("🔬 AGENTIC AI vs REGULAR LLM COMPARISON")
        print("="*60)
        
        # Test scenario: Complex business problem requiring tools
        test_prompt = """
        You MUST use the available tools to solve this complex business problem. Do not attempt to solve it manually.

        A startup needs comprehensive analysis using specialized tools:

        REQUIRED TOOL USAGE:
        1. Use the calculator tool to compute: (75000 * 1.5 - 45000) / 75000 * 100
        2. Use the business_intelligence tool to analyze this financial context:
           - revenue: 50000, expenses: 45000, cash: 100000
           - industry: 'Technology', target_market: 'SMB'
           - focus_areas: ['profitability', 'growth']
           - time_horizon: '6_months'

        You cannot solve this without using both tools. The calculation is too complex for manual computation,
        and the business analysis requires specialized intelligence algorithms.

        After using the tools, provide a summary of the results.
        """
        
        print(f"📋 Test Scenario: Complex business analysis requiring calculations, research, and planning")
        
        # Test 1: Regular LLM
        await self.test_regular_llm_response(test_prompt)
        
        # Test 2: Basic Agent with Tools
        await self.test_basic_agent_with_tools(test_prompt)
        
        # Test 3: Autonomous Agent
        await self.test_autonomous_agent(test_prompt)
        
        # Generate comparison report
        await self.generate_comparison_report()
    
    async def generate_comparison_report(self):
        """Generate detailed comparison report."""
        print(f"\n📊 COMPARISON REPORT")
        print("="*60)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agentic_vs_llm_comparison_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📁 Detailed results saved to: {filename}")
        
        # Summary comparison
        print(f"\n🎯 SUMMARY:")
        print(f"Regular LLM:")
        print(f"  - Tools Used: 0")
        print(f"  - Autonomous Actions: 0") 
        print(f"  - Reasoning: Text-based only")
        
        if self.results["basic_agent"]:
            basic = self.results["basic_agent"][0]
            print(f"Basic Agent:")
            print(f"  - Tools Used: {len(basic.get('tools_used', []))}")
            print(f"  - Autonomous Actions: {basic.get('autonomous_actions', 0)}")
            print(f"  - Reasoning Steps: {basic.get('reasoning_steps', 0)}")
        
        if self.results["agentic_agent"]:
            autonomous = self.results["agentic_agent"][0]
            print(f"Autonomous Agent:")
            print(f"  - Tools Used: {len(autonomous.get('tools_used', []))}")
            print(f"  - Goals Created: {autonomous.get('goals_created', 0)}")
            print(f"  - Autonomous Actions: {autonomous.get('autonomous_actions', 0)}")
            print(f"  - Learning Enabled: {autonomous.get('learning_enabled', False)}")

async def main():
    """Run the comparison test."""
    comparison = AgentBehaviorComparison()
    await comparison.run_comparison_test()

if __name__ == "__main__":
    asyncio.run(main())
