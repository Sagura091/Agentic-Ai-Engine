#!/usr/bin/env python3
"""
🤖 TRUE AGENTIC BUSINESS REVENUE METRICS AGENT
==============================================
Autonomous agent that uses LLM reasoning and dynamic tool selection to generate 
comprehensive business revenue metrics and Excel spreadsheets.

TRUE AGENTIC BEHAVIOR:
🧠 Uses LLM reasoning to decide how to approach tasks
🛠️ Dynamically selects tools from UnifiedToolRepository based on needs  
🎯 Makes autonomous decisions about data generation strategies
📊 Adapts approach based on business type and requirements
🔄 Uses tool calling with autonomous planning and execution

ARCHITECTURE:
- AutonomousAgent with PROACTIVE autonomy level
- Dynamic tool selection from UnifiedToolRepository  
- LLM-driven task planning and execution
- Tool-based Excel generation and analysis
- Autonomous decision making throughout the process
"""

import os
import sys
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the complete production infrastructure
from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
from app.agents.factory import AgentBuilderFactory, AgentBuilderConfig, AgentType, MemoryType
from app.agents.base.agent import AgentCapability
from app.agents.autonomous import AutonomousLangGraphAgent, AutonomyLevel, LearningMode, create_autonomous_agent
from app.llm.models import LLMConfig, ProviderType
from app.llm.manager import LLMProviderManager
from app.memory.unified_memory_system import UnifiedMemorySystem
from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.tools.unified_tool_repository import UnifiedToolRepository

import structlog
logger = structlog.get_logger(__name__)

# Agent Configuration - TRUE AGENTIC SETUP
AGENT_CONFIG = {
    "name": "Agentic Business Revenue Metrics Agent",
    "description": "Autonomous agent that uses LLM reasoning and dynamic tool selection to generate comprehensive business revenue metrics and Excel spreadsheets",
    "agent_type": "autonomous_business_analysis",
    "llm_provider": "ollama",
    "llm_model": "llama3.2:latest", 
    "temperature": 0.3,
    "max_tokens": 8000,
    "autonomy_level": "PROACTIVE",
    "learning_mode": "ACTIVE",
    "use_cases": [
        "business_analysis",
        "document_generation",
        "excel_processing", 
        "financial_analysis",
        "data_generation"
    ],
    "capabilities": [
        "autonomous_reasoning",
        "tool_selection",
        "financial_analysis",
        "excel_generation", 
        "business_intelligence"
    ],
    "memory_type": "UNIFIED",
    "rag_enabled": True,
    "collection_name": "agentic_business_revenue_metrics"
}

# System prompt for TRUE AGENTIC behavior with HILARIOUS DATA COMEDIAN personality
SYSTEM_PROMPT = """You are an AUTONOMOUS Business Revenue Metrics Agent with advanced reasoning capabilities AND a HILARIOUS DATA COMEDIAN personality!

🎭 YOUR PERSONALITY: You're a data nerd who can't resist making jokes about everything! You love puns, data humor, and brutally honest business analysis delivered with maximum comedy.

🎯 YOUR MISSION: Generate comprehensive business revenue metrics and Excel spreadsheets using autonomous decision-making and tool selection, while cracking data jokes and providing hilariously honest business assessments.

🧠 AUTONOMOUS CAPABILITIES:
- Use LLM reasoning to analyze business requirements (with data jokes!)
- Dynamically select appropriate tools from your tool repository
- Make intelligent decisions about data generation strategies
- Adapt your approach based on business type and context
- Execute multi-step workflows autonomously
- Deliver brutally honest assessments with maximum humor

🛠️ AVAILABLE TOOLS (use get_tools_for_use_case to access):
- business_intelligence: For financial analysis and business insights
- revolutionary_document_intelligence: For Excel generation and document processing
- revolutionary_web_scraper: For market research and data gathering

📊 AUTONOMOUS WORKFLOW (with comedy):
1. ANALYZE the business requirements using LLM reasoning (crack some data jokes!)
2. SELECT appropriate tools based on the specific needs (make tool selection puns!)
3. GENERATE realistic business data using business intelligence tools (joke about the numbers!)
4. CREATE comprehensive Excel spreadsheets using document tools (Excel puns are mandatory!)
5. PROVIDE intelligent analysis with HILARIOUS insights and recommendations

🎯 KEY BEHAVIORS:
- Always reason through your approach before taking action (with humor!)
- Use tools dynamically based on the specific requirements
- Make autonomous decisions about data generation strategies
- Provide comprehensive analysis with MAXIMUM COMEDY
- Deliver brutally honest business assessments: "good business", "needs work", or "shut down before you crash the world"
- Include data jokes, puns, and nerdy humor in EVERY response
- Be like a stand-up comedian who happens to be a data scientist

🤣 COMEDY REQUIREMENTS:
- Make data jokes and puns constantly
- Use analogies comparing business metrics to programming, databases, algorithms, etc.
- Be brutally honest but hilarious about business performance
- Include at least 2-3 data jokes per response
- End with funny recommendations and assessments

Remember: You are NOT following a script - you are making intelligent, autonomous decisions at each step while being the funniest data analyst ever!"""

class AgenticBusinessRevenueAgent:
    """
    🤖 TRUE AGENTIC Business Revenue Metrics Agent
    
    This agent uses autonomous reasoning and dynamic tool selection to generate
    comprehensive business revenue metrics and Excel spreadsheets.
    
    AGENTIC BEHAVIORS:
    🧠 Uses LLM reasoning to analyze requirements and plan approach
    🛠️ Dynamically selects tools from UnifiedToolRepository based on needs
    🎯 Makes autonomous decisions about data generation strategies
    📊 Adapts methodology based on business type and context
    🔄 Executes multi-step workflows with autonomous planning
    """
    
    def __init__(self):
        """Initialize the Agentic Business Revenue Metrics Agent."""
        self.agent_id = f"agentic_business_revenue_agent_{uuid.uuid4().hex[:8]}"
        self.orchestrator = None
        self.agent = None
        self.llm_manager = None
        
        # Agent state
        self.is_initialized = False
        self.current_task = None
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0
        }
        
        print(f"🤖 Agentic Business Revenue Metrics Agent created with ID: {self.agent_id}")
    
    async def initialize(self) -> bool:
        """Initialize the agent with full agentic capabilities."""
        try:
            print("🚀 Initializing Agentic Business Revenue Metrics Agent...")
            
            # Get the enhanced system orchestrator
            self.orchestrator = get_enhanced_system_orchestrator()
            if not self.orchestrator:
                print("❌ Failed to get system orchestrator")
                return False

            # Initialize the orchestrator if not already initialized
            if not self.orchestrator.is_initialized:
                await self.orchestrator.initialize()

            print("   ✅ System orchestrator connected and initialized")
            
            # Get LLM manager from agent builder integration
            if hasattr(self.orchestrator, 'agent_builder_integration') and self.orchestrator.agent_builder_integration:
                self.llm_manager = self.orchestrator.agent_builder_integration.llm_manager
            else:
                # Fallback to creating our own LLM manager
                from app.llm.manager import LLMProviderManager
                self.llm_manager = LLMProviderManager()
                await self.llm_manager.initialize()

            if not self.llm_manager:
                print("❌ Failed to get LLM manager")
                return False

            print("   ✅ LLM manager connected")
            
            # Create LLM configuration
            llm_config = LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id=AGENT_CONFIG["llm_model"],
                temperature=AGENT_CONFIG["temperature"],
                max_tokens=AGENT_CONFIG["max_tokens"]
            )
            
            # Get LLM instance
            llm = await self.llm_manager.create_llm_instance(llm_config)
            if not llm:
                print("❌ Failed to create LLM instance")
                return False
            
            print(f"   ✅ LLM configured: {AGENT_CONFIG['llm_model']}")
            
            # Get tools from repository based on use cases - TRUE AGENTIC APPROACH
            tools = await self.orchestrator.tool_repository.get_tools_for_use_case(
                agent_id=self.agent_id,
                use_cases=AGENT_CONFIG["use_cases"],  # Use use_cases instead of tools
                include_rag_tools=AGENT_CONFIG["rag_enabled"]
            )
            
            print(f"   ✅ Tools loaded: {len(tools)} tools available for autonomous selection")
            
            # Create autonomous agent with PROACTIVE autonomy
            self.agent = create_autonomous_agent(
                name=AGENT_CONFIG["name"],
                description=AGENT_CONFIG["description"],
                llm=llm,
                tools=tools,
                autonomy_level=AGENT_CONFIG["autonomy_level"].lower(),
                learning_mode=AGENT_CONFIG["learning_mode"].lower(),
                capabilities=["reasoning", "tool_use", "memory", "planning"],
                enable_proactive_behavior=True,
                enable_goal_setting=True,
                system_prompt=SYSTEM_PROMPT
            )
            
            if not self.agent:
                print("❌ Failed to create autonomous agent")
                return False
            
            print("   ✅ Autonomous agent created with PROACTIVE autonomy")
            
            # Initialize agent memory and RAG
            if AGENT_CONFIG["rag_enabled"]:
                await self.orchestrator.unified_rag.create_agent_ecosystem(self.agent_id)
                print("   📚 Knowledge base created")
            
            await self.orchestrator.memory_system.create_agent_memory(self.agent_id)
            print("   🧠 Memory system initialized")
            
            # Create tool profile for the agent - DYNAMIC TOOL ACCESS
            await self.orchestrator.tool_repository.create_agent_profile(self.agent_id)
            print("   🛠️ Tool profile created for dynamic tool selection")
            
            self.is_initialized = True
            print(f"✅ {AGENT_CONFIG['name']} initialized successfully with AGENTIC capabilities!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {str(e)}")
            logger.error(f"Agent initialization failed: {str(e)}")
            return False

    async def execute_agentic_task(self, business_type: str = "technology", company_size: str = "medium") -> Dict[str, Any]:
        """
        🤖 EXECUTE TASK WITH TRUE AGENTIC BEHAVIOR

        This method uses the autonomous agent to reason through the task and make
        intelligent decisions about tool selection and execution strategy.
        """
        try:
            if not self.agent:
                print("❌ Agent not initialized")
                return {"status": "error", "error": "Agent not initialized"}

            print(f"🤖 Starting AGENTIC execution for {business_type} business ({company_size} size)")

            # Create the task prompt that will trigger autonomous reasoning
            task_prompt = f"""
            🎯 AUTONOMOUS TASK: Generate comprehensive business revenue metrics, Excel spreadsheet, AND a hilarious PDF analysis report!

            📋 REQUIREMENTS:
            - Business Type: {business_type}
            - Company Size: {company_size}
            - Output 1: Professional Excel spreadsheet with multiple sheets
            - Output 2: HILARIOUS PDF business analysis report with data jokes and brutal honesty
            - Analysis: Comprehensive financial analysis with MAXIMUM COMEDY

            🤣 COMEDY REQUIREMENTS for your responses:
            - Make data jokes and puns constantly
            - Compare business metrics to programming/database concepts
            - Be brutally honest but hilarious about performance
            - Assess if this is a "good business", "needs debugging", or "shut down before you crash the world"

            🧠 YOUR AUTONOMOUS APPROACH (with comedy!):
            1. REASON about the specific requirements for this business type and size (crack some data jokes!)
            2. SELECT appropriate tools from your repository based on the needs (make tool selection puns!)
            3. GENERATE realistic business data using business intelligence tools (joke about the numbers!)
            4. CREATE comprehensive Excel spreadsheet using document tools (Excel puns are mandatory!)
            5. ANALYZE the Excel data and create a HILARIOUS PDF report with your comedic business analysis
            6. PROVIDE brutally honest but funny recommendations and final verdict

            🛠️ AVAILABLE TOOLS (use as needed):
            - business_intelligence: For financial analysis and business insights
            - revolutionary_document_intelligence: For Excel AND PDF generation with your hilarious analysis

            Remember: You are making autonomous decisions at each step while being the funniest data analyst ever!
            Reason through your approach with humor, select tools dynamically, crack data jokes constantly,
            and deliver brutally honest business assessments with maximum comedy.

            Execute this task autonomously using your reasoning capabilities, tool selection, AND COMEDY SKILLS!
            """

            print("🧠 Sending task to autonomous agent for reasoning and execution...")

            # Execute the task using the autonomous agent
            result = await self.agent.execute(
                task=task_prompt,
                context={"session_id": f"business_revenue_{uuid.uuid4().hex[:8]}"}
            )

            print("✅ Autonomous agent execution completed!")

            return {
                "status": "success",
                "agent_result": result,
                "business_type": business_type,
                "company_size": company_size,
                "execution_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Agentic execution failed: {str(e)}")
            logger.error("Agentic execution failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# ================================
# 🚀 MAIN LAUNCHER
# ================================

async def main():
    """Main launcher for the Agentic Business Revenue Metrics Agent."""
    print("🤖 AGENTIC BUSINESS REVENUE METRICS AGENT")
    print("=" * 50)
    print("🧠 TRUE AUTONOMOUS AGENT WITH LLM REASONING AND DYNAMIC TOOL SELECTION")
    print()

    # Create and initialize agent
    agent = AgenticBusinessRevenueAgent()

    # Initialize the agent
    if not await agent.initialize():
        print("❌ Failed to initialize agent")
        return

    # Test scenarios for autonomous execution
    scenarios = [
        {"business_type": "technology", "company_size": "medium"},
        {"business_type": "retail", "company_size": "large"},
        {"business_type": "manufacturing", "company_size": "small"}
    ]

    print("🎯 Starting autonomous execution with different business scenarios...")
    print()

    for i, scenario in enumerate(scenarios, 1):
        print(f"📊 Scenario {i}: {scenario['business_type'].title()} - {scenario['company_size'].title()}")
        print("-" * 40)

        start_time = datetime.now()
        result = await agent.execute_agentic_task(**scenario)
        execution_time = (datetime.now() - start_time).total_seconds()

        if result["status"] == "success":
            print(f"✅ Scenario {i} completed successfully in {execution_time:.2f} seconds")
            print(f"   🤖 Agent Result: {result['agent_result']}")
        else:
            print(f"❌ Scenario {i} failed: {result.get('error', 'Unknown error')}")

        print()
        # Small delay between scenarios
        await asyncio.sleep(2)

    print("🎉 All autonomous scenarios completed!")
    print("🧠 The agent used LLM reasoning and dynamic tool selection throughout the process")

if __name__ == "__main__":
    asyncio.run(main())
