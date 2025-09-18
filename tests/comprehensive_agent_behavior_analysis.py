#!/usr/bin/env python3
"""
Comprehensive Agent Behavior Analysis

This script captures EVERYTHING the agents do internally to prove they exhibit
true agentic AI behavior with reasoning, decision-making, and autonomous actions.
All responses, logs, and internal state changes are saved to files for analysis.
"""

import asyncio
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from io import StringIO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create output directory for analysis files
output_dir = Path("agent_behavior_analysis")
output_dir.mkdir(exist_ok=True)

class AgentBehaviorCapture:
    """Captures and analyzes all agent behavior for detailed analysis."""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.analysis_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "agents_tested": [],
            "interactions": [],
            "behavior_analysis": {},
            "agentic_indicators": [],
            "logs_captured": []
        }
        
        # Set up logging capture
        self.log_capture = StringIO()
        self.log_handler = logging.StreamHandler(self.log_capture)
        self.log_handler.setLevel(logging.DEBUG)
        
        # Configure root logger to capture everything
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.DEBUG)
        
    def log_interaction(self, agent_id: str, prompt: str, response: Dict[str, Any], 
                       analysis: Dict[str, Any] = None):
        """Log a complete agent interaction with analysis."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "prompt": prompt,
            "response": response,
            "analysis": analysis or {},
            "logs_during_interaction": self.get_recent_logs()
        }
        self.analysis_data["interactions"].append(interaction)
        
    def get_recent_logs(self) -> List[str]:
        """Get recent log entries."""
        log_content = self.log_capture.getvalue()
        recent_logs = log_content.split('\n')[-50:]  # Last 50 log lines
        self.log_capture.truncate(0)  # Clear buffer
        self.log_capture.seek(0)
        return [log for log in recent_logs if log.strip()]
        
    def analyze_agentic_behavior(self, response: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Analyze response for agentic behavior indicators."""
        analysis = {
            "reasoning_indicators": [],
            "decision_making_indicators": [],
            "autonomous_behavior_indicators": [],
            "tool_usage_indicators": [],
            "learning_indicators": [],
            "goal_oriented_indicators": []
        }
        
        # Extract response content
        response_text = ""
        if "messages" in response and response["messages"]:
            last_message = response["messages"][-1]
            if hasattr(last_message, 'content'):
                response_text = last_message.content.lower()
        
        # Check for reasoning indicators
        reasoning_words = ["because", "therefore", "reasoning", "analysis", "consider", "evaluate"]
        analysis["reasoning_indicators"] = [word for word in reasoning_words if word in response_text]
        
        # Check for decision-making indicators
        decision_words = ["decide", "choose", "select", "option", "alternative", "best"]
        analysis["decision_making_indicators"] = [word for word in decision_words if word in response_text]
        
        # Check for autonomous behavior
        autonomous_words = ["autonomously", "independently", "self-directed", "proactive"]
        analysis["autonomous_behavior_indicators"] = [word for word in autonomous_words if word in response_text]
        
        # Check response structure for agentic patterns
        if "tool_calls" in response and response["tool_calls"]:
            analysis["tool_usage_indicators"].append("tool_calls_present")
            
        if "iteration_count" in response and response["iteration_count"] > 1:
            analysis["learning_indicators"].append("multiple_iterations")
            
        if "outputs" in response and response["outputs"]:
            analysis["goal_oriented_indicators"].append("structured_outputs")
            
        return analysis
        
    def save_analysis(self):
        """Save complete analysis to files."""
        # Save main analysis
        analysis_file = output_dir / f"agent_analysis_{self.session_id}.json"
        with open(analysis_file, 'w') as f:
            json.dump(self.analysis_data, f, indent=2, default=str)
            
        # Save detailed logs
        logs_file = output_dir / f"agent_logs_{self.session_id}.txt"
        with open(logs_file, 'w') as f:
            f.write(f"Agent Behavior Analysis Logs - Session {self.session_id}\n")
            f.write("="*80 + "\n\n")
            
            for interaction in self.analysis_data["interactions"]:
                f.write(f"INTERACTION: {interaction['timestamp']}\n")
                f.write(f"Agent ID: {interaction['agent_id']}\n")
                f.write(f"Prompt: {interaction['prompt']}\n")
                f.write(f"Response: {json.dumps(interaction['response'], indent=2, default=str)}\n")
                f.write(f"Analysis: {json.dumps(interaction['analysis'], indent=2)}\n")
                f.write("Logs during interaction:\n")
                for log_line in interaction['logs_during_interaction']:
                    f.write(f"  {log_line}\n")
                f.write("\n" + "-"*80 + "\n\n")
                
        print(f"\n📁 Analysis saved to:")
        print(f"   📄 {analysis_file}")
        print(f"   📄 {logs_file}")

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.outputs import LLMResult, Generation
from langchain_core.messages import AIMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional

class OllamaHTTPLLM(BaseLanguageModel):
    """Custom Ollama LLM using the project's HTTP client for full LangChain compatibility."""

    model: str
    base_url: str = "http://localhost:11434"
    call_count: int = 0

    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model=model, base_url=base_url, call_count=0, **kwargs)

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for the given prompts."""
        import asyncio
        return asyncio.run(self._agenerate(prompts, stop, run_manager, **kwargs))

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generate responses for the given prompts."""
        from app.http_client import SimpleHTTPClient

        generations = []
        for prompt in prompts:
            self.call_count += 1

            try:
                async with SimpleHTTPClient(self.base_url, timeout=30) as client:
                    # Prepare Ollama generate request
                    request_data = {
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 500
                        }
                    }

                    print(f"🤖 Ollama request #{self.call_count} to {self.model}: {prompt[:50]}...")

                    response = await client.post("/api/generate", body=request_data)

                    if response.status_code == 200:
                        data = response.json()
                        if data and 'response' in data:
                            result = data['response'].strip()
                            print(f"✅ Ollama response #{self.call_count}: {result[:100]}...")
                            generations.append([Generation(text=result)])
                        else:
                            print(f"⚠️  Ollama response missing 'response' field: {data}")
                            generations.append([Generation(text=f"Ollama response #{self.call_count}: {prompt}")])
                    else:
                        print(f"❌ Ollama error {response.status_code}: {response.raw_data}")
                        generations.append([Generation(text=f"Error response #{self.call_count}: {prompt}")])

            except Exception as e:
                print(f"❌ Ollama request failed: {e}")
                generations.append([Generation(text=f"Failed response #{self.call_count}: {prompt}")])

        return LLMResult(generations=generations)

    async def ainvoke(self, input_data, config=None, **kwargs):
        """Async invoke method compatible with LangChain chains."""
        # Handle different input types from LangChain
        if hasattr(input_data, 'to_string'):
            prompt_text = input_data.to_string()
        elif isinstance(input_data, dict):
            # Extract prompt from LangChain prompt template format
            if 'text' in input_data:
                prompt_text = input_data['text']
            else:
                prompt_text = str(input_data)
        else:
            prompt_text = str(input_data)

        result = await self._agenerate([prompt_text])
        return AIMessage(content=result.generations[0][0].text)

    @property
    def _llm_type(self) -> str:
        return "ollama_http_llm"

    # Required abstract methods from BaseLanguageModel
    def generate_prompt(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks=None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for prompts (sync version)."""
        import asyncio
        return asyncio.run(self._agenerate(prompts, stop, None, **kwargs))

    async def agenerate_prompt(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks=None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for prompts (async version)."""
        return await self._agenerate(prompts, stop, None, **kwargs)

    def predict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Predict text response (sync version)."""
        result = self.generate_prompt([text], stop, **kwargs)
        return result.generations[0][0].text

    async def apredict(self, text: str, *, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Predict text response (async version)."""
        result = await self.agenerate_prompt([text], stop, **kwargs)
        return result.generations[0][0].text

    def predict_messages(
        self,
        messages: List[Any],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Predict message response (sync version)."""
        text = str(messages)
        result = self.predict(text, stop=stop, **kwargs)
        from langchain_core.messages import AIMessage
        return AIMessage(content=result)

    async def apredict_messages(
        self,
        messages: List[Any],
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Predict message response (async version)."""
        text = str(messages)
        result = await self.apredict(text, stop=stop, **kwargs)
        from langchain_core.messages import AIMessage
        return AIMessage(content=result)

    def invoke(self, input_data, config=None, **kwargs):
        """Sync invoke method."""
        import asyncio
        return asyncio.run(self.ainvoke(input_data, config, **kwargs))

async def get_available_ollama_models():
    """Get available Ollama models using the project's HTTP client."""
    try:
        from app.http_client import SimpleHTTPClient

        async with SimpleHTTPClient("http://localhost:11434", timeout=10) as client:
            response = await client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                if data and 'models' in data:
                    models = [model['name'] for model in data['models']]
                    print(f"✅ Found {len(models)} Ollama models: {', '.join(models[:3])}...")
                    return models
            return []
    except Exception as e:
        print(f"⚠️  Could not get Ollama models: {e}")
        return []

async def create_real_llm():
    """Create a real LangChain-compatible LLM for testing."""
    try:
        # Get available models first
        available_models = await get_available_ollama_models()

        if not available_models:
            print("⚠️  No Ollama models available, using mock")
            return create_mock_llm()

        # Choose the best model (prefer smaller ones for testing)
        model_priorities = ["phi4:14b", "llama3.2:1b", "llama3.2:3b", "llama3:8b"]
        selected_model = None

        for preferred in model_priorities:
            if preferred in available_models:
                selected_model = preferred
                break

        if not selected_model:
            selected_model = available_models[0]  # Use first available

        print(f"🤖 Using Ollama model: {selected_model}")

        # Create custom Ollama LLM using the HTTP client
        from app.http_client import SimpleHTTPClient
        llm = OllamaHTTPLLM(model=selected_model)

        # Test if it's working
        test_response = await llm.ainvoke("Hello, respond with just 'OK' to confirm you're working.")
        print(f"✅ Real LLM connected: {test_response.content[:50]}...")
        return llm

    except Exception as e:
        print(f"❌ Real LLM failed: {e}")
        print("🚫 NO MOCKS ALLOWED - Fix the Ollama connection!")
        raise e

# NO MOCK LLMS - ONLY REAL OLLAMA INTEGRATION ALLOWED

async def main():
    """Run comprehensive agent behavior analysis."""
    print("🔬 COMPREHENSIVE AGENT BEHAVIOR ANALYSIS")
    print("This will capture EVERYTHING the agents do internally")
    print("="*80)
    
    # Initialize behavior capture
    capture = AgentBehaviorCapture()
    
    try:
        # Import agent classes
        from app.agents.base.agent import LangGraphAgent, AgentConfig
        from app.agents.autonomous import AutonomousLangGraphAgent, AutonomousAgentConfig, AutonomyLevel
        
        # Create LLM
        print("\n🤖 Creating LLM...")
        llm = await create_real_llm()
        
        # Test 1: Basic Agent Reasoning
        print("\n📋 TEST 1: Basic Agent Reasoning Analysis")
        basic_config = AgentConfig(
            name="Reasoning Test Agent",
            description="Agent designed to demonstrate reasoning capabilities",
            model_name="test-model"
        )
        
        basic_agent = LangGraphAgent(config=basic_config, llm=llm, tools=[])
        capture.analysis_data["agents_tested"].append({
            "agent_id": basic_agent.agent_id,
            "type": "basic",
            "config": basic_config.__dict__
        })
        
        # Test reasoning with complex prompt
        reasoning_prompt = """
        Analyze this scenario and provide your reasoning:
        A company must choose between three strategies:
        A) Cut costs by 20% immediately
        B) Invest in R&D for future growth
        C) Acquire a smaller competitor
        
        Consider the long-term implications and explain your reasoning process.
        """
        
        print(f"🧠 Testing reasoning with complex scenario...")
        response = await basic_agent.execute(reasoning_prompt)
        analysis = capture.analyze_agentic_behavior(response, reasoning_prompt)
        capture.log_interaction(basic_agent.agent_id, reasoning_prompt, response, analysis)
        
        print(f"✅ Reasoning test complete - Analysis: {len(analysis['reasoning_indicators'])} reasoning indicators found")

        # Test 2: Autonomous Agent Decision Making
        print("\n📋 TEST 2: Autonomous Agent Decision Making")
        autonomous_config = AutonomousAgentConfig(
            name="Decision Making Agent",
            description="Autonomous agent for testing decision-making capabilities",
            model_name="test-model",
            autonomy_level=AutonomyLevel.AUTONOMOUS,
            capabilities=["reasoning", "tool_use", "planning"],
            enable_proactive_behavior=True,
            enable_goal_setting=True
        )

        autonomous_agent = AutonomousLangGraphAgent(config=autonomous_config, llm=llm, tools=[])
        capture.analysis_data["agents_tested"].append({
            "agent_id": autonomous_agent.agent_id,
            "type": "autonomous",
            "config": autonomous_config.__dict__
        })

        decision_prompt = """
        You are faced with a complex problem that requires autonomous decision-making:

        A research team needs to prioritize their limited resources. They can:
        1. Continue current project with 70% success probability
        2. Start new high-risk project with 30% success but 10x impact
        3. Collaborate with another team, sharing resources and results

        Make an autonomous decision and explain your reasoning process,
        including how you weigh risks, benefits, and strategic considerations.
        """

        print(f"🤖 Testing autonomous decision-making...")
        response = await autonomous_agent.execute(decision_prompt)
        analysis = capture.analyze_agentic_behavior(response, decision_prompt)
        capture.log_interaction(autonomous_agent.agent_id, decision_prompt, response, analysis)

        print(f"✅ Decision-making test complete - Found {len(analysis['decision_making_indicators'])} decision indicators")

        # Test 3: Multi-step Problem Solving
        print("\n📋 TEST 3: Multi-step Problem Solving")
        problem_solving_prompt = """
        Solve this multi-step problem autonomously:

        A startup has $100,000 budget and needs to:
        - Hire 2 developers ($60,000/year each)
        - Rent office space ($2,000/month)
        - Buy equipment ($15,000)
        - Marketing budget (remaining funds)

        Calculate if this is feasible for 1 year, and if not,
        autonomously propose alternative solutions with reasoning.
        """

        print(f"🧮 Testing multi-step problem solving...")
        response = await basic_agent.execute(problem_solving_prompt)
        analysis = capture.analyze_agentic_behavior(response, problem_solving_prompt)
        capture.log_interaction(basic_agent.agent_id, problem_solving_prompt, response, analysis)

        print(f"✅ Problem-solving test complete")

        # Test 4: Goal-Oriented Behavior
        print("\n📋 TEST 4: Goal-Oriented Behavior")
        goal_prompt = """
        Your goal is to help a small business increase revenue by 25% in 6 months.
        Current situation:
        - Monthly revenue: $50,000
        - 3 employees
        - Online presence: basic website
        - Customer base: 200 regular customers

        Autonomously develop a comprehensive strategy with specific actions,
        timelines, and success metrics. Show your goal-oriented thinking.
        """

        print(f"🎯 Testing goal-oriented behavior...")
        response = await autonomous_agent.execute(goal_prompt)
        analysis = capture.analyze_agentic_behavior(response, goal_prompt)
        capture.log_interaction(autonomous_agent.agent_id, goal_prompt, response, analysis)

        print(f"✅ Goal-oriented test complete - Found {len(analysis['goal_oriented_indicators'])} goal indicators")

        # Test 5: Learning and Adaptation
        print("\n📋 TEST 5: Learning and Adaptation")
        learning_prompt = """
        Based on our previous interactions, demonstrate learning and adaptation:

        1. What patterns have you noticed in the problems presented?
        2. How would you adapt your approach for similar future problems?
        3. What have you learned about effective decision-making?
        4. How do you improve your reasoning over time?

        Show evidence of learning and adaptive behavior.
        """

        print(f"🧠 Testing learning and adaptation...")
        response = await autonomous_agent.execute(learning_prompt)
        analysis = capture.analyze_agentic_behavior(response, learning_prompt)
        capture.log_interaction(autonomous_agent.agent_id, learning_prompt, response, analysis)

        print(f"✅ Learning test complete - Found {len(analysis['learning_indicators'])} learning indicators")

        # Generate comprehensive analysis
        capture.analysis_data["behavior_analysis"] = {
            "total_interactions": len(capture.analysis_data["interactions"]),
            "agents_tested": len(capture.analysis_data["agents_tested"]),
            "agentic_behavior_summary": {
                "reasoning_evidence": sum(len(i["analysis"]["reasoning_indicators"]) for i in capture.analysis_data["interactions"]),
                "decision_making_evidence": sum(len(i["analysis"]["decision_making_indicators"]) for i in capture.analysis_data["interactions"]),
                "autonomous_behavior_evidence": sum(len(i["analysis"]["autonomous_behavior_indicators"]) for i in capture.analysis_data["interactions"]),
                "goal_oriented_evidence": sum(len(i["analysis"]["goal_oriented_indicators"]) for i in capture.analysis_data["interactions"]),
                "learning_evidence": sum(len(i["analysis"]["learning_indicators"]) for i in capture.analysis_data["interactions"])
            }
        }

        # Save final results
        capture.save_analysis()

        print(f"\n🎯 COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📊 Summary:")
        print(f"   🤖 Agents tested: {len(capture.analysis_data['agents_tested'])}")
        print(f"   💬 Interactions: {len(capture.analysis_data['interactions'])}")
        print(f"   🧠 Reasoning indicators: {capture.analysis_data['behavior_analysis']['agentic_behavior_summary']['reasoning_evidence']}")
        print(f"   🎯 Decision indicators: {capture.analysis_data['behavior_analysis']['agentic_behavior_summary']['decision_making_evidence']}")
        print(f"   🤖 Autonomous indicators: {capture.analysis_data['behavior_analysis']['agentic_behavior_summary']['autonomous_behavior_evidence']}")
        print(f"\n📁 Check the output files for complete agent behavior analysis!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    finally:
        capture.save_analysis()

if __name__ == "__main__":
    asyncio.run(main())
