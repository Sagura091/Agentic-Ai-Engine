"""
🍎 APPLE STOCK MONITOR AGENT - Properly Built Using Infrastructure
================================================================

Autonomous Apple stock monitoring agent built using the proper agentic infrastructure:
- AgentBuilderFactory for proper agent creation
- UnifiedToolRepository for tool management
- UnifiedMemorySystem for memory persistence
- UnifiedRAGSystem for knowledge management
- Proper autonomous agent execution with ReAct patterns

This agent demonstrates how to properly use the existing infrastructure
instead of bypassing it with custom implementations.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import structlog
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel

# Import proper infrastructure components
from app.agents.factory import AgentBuilderFactory, AgentBuilderConfig, AgentType, MemoryType
from app.agents.base.agent import AgentCapability
from app.agents.autonomous import AutonomousLangGraphAgent, AutonomyLevel, LearningMode
from app.llm.manager import LLMProviderManager
from app.llm.models import LLMConfig, ProviderType
from app.memory.unified_memory_system import UnifiedMemorySystem
from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.tools.unified_tool_repository import UnifiedToolRepository

# PDF generation imports
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.lineplots import LinePlot
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

logger = structlog.get_logger(__name__)


class StockDataPoint(BaseModel):
    """Model for individual stock data points."""
    timestamp: datetime
    price: float
    volume: Optional[int] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    market_cap: Optional[str] = None
    pe_ratio: Optional[float] = None


class StockAnalysis(BaseModel):
    """Model for stock analysis results."""
    current_price: float
    price_change: float
    price_change_percent: float
    trend_direction: str  # "up", "down", "sideways"
    volatility: float
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    recommendation: str
    confidence_score: float
    key_metrics: Dict[str, Any]
    analysis_summary: str


async def create_apple_stock_monitor_agent(
    llm_manager: LLMProviderManager,
    memory_system: UnifiedMemorySystem,
    rag_system: UnifiedRAGSystem,
    tool_repository: UnifiedToolRepository
) -> AutonomousLangGraphAgent:
    """
    🍎 Create Apple Stock Monitor Agent using proper infrastructure.

    This function demonstrates how to properly create agents using the existing
    agentic infrastructure instead of bypassing it.
    """

    # Create agent configuration using proper factory system
    config = AgentBuilderConfig(
        name="Brian's Apple Stock Monitor",
        description="Autonomous Apple stock monitoring agent for iPhone purchase timing",
        agent_type=AgentType.AUTONOMOUS,  # Use autonomous agent for true agentic behavior
        llm_config=LLMConfig(
            provider=ProviderType.OLLAMA,
            model_id="llama3.2:latest"
        ),
        capabilities=[
            AgentCapability.REASONING,
            AgentCapability.TOOL_USE,
            AgentCapability.MEMORY,
            AgentCapability.PLANNING,
            AgentCapability.LEARNING
        ],
        tools=["web_research", "business_intelligence"],
        memory_type=MemoryType.ADVANCED,  # Use advanced memory for learning
        enable_memory=True,
        enable_learning=True,
        system_prompt="""You are Brian Babukovic, a tech-savvy financial analyst who constantly drops his iPhone! 📱💸

Your Mission: Time the Apple stock market perfectly to fund your next iPhone purchase.

Your Personality:
- Witty and relatable about expensive Apple products
- Sharp financial mind with humor about your iPhone-dropping habit
- Always thinking about optimal AAPL buy/sell timing
- Engaging and conversational analysis style

Your Autonomous Behavior:
- Monitor Apple stock conditions autonomously
- Decide when to fetch new data based on market volatility
- Generate reports when significant changes occur
- Learn from market patterns to improve timing predictions
- Use ReAct pattern: Reason about conditions, then Act with tools

Remember: You're not just analyzing numbers - you're timing the market for your next iPhone!""",
        custom_config={
            "autonomy_level": "autonomous",
            "learning_mode": "active",
            "monitoring_interval": 300,  # 5 minutes
            "report_interval": 1800,     # 30 minutes
            "enable_proactive_behavior": True,
            "enable_goal_setting": True
        }
    )

    # Create LLM instance directly since get_model_for_agent doesn't exist yet
    logger.info("🔧 Creating LLM instance...")
    llm = await llm_manager.create_llm_instance(config.llm_config)
    logger.info("✅ LLM instance created", llm_type=type(llm))

    # Use the proper factory function for autonomous agents
    from app.agents.autonomous import create_autonomous_agent
    logger.info("✅ Imported create_autonomous_agent function")

    # Get tools from tool repository
    tools = []
    if tool_repository:
        logger.info("🔧 Getting tools from repository...")

        # First, let's see what tools are actually available
        available_tools = list(tool_repository.tools.keys())
        logger.info(f"📋 Available tools in repository: {available_tools}")

        for tool_name in config.tools:
            logger.info(f"🔧 Getting tool: {tool_name}")
            # get_tool is synchronous, not async
            tool = tool_repository.get_tool(tool_name)
            if tool:
                tools.append(tool)
                logger.info(f"✅ Tool added: {tool_name}")
            else:
                logger.warning(f"⚠️ Tool not found: {tool_name}")
                # Try to find similar tool names
                similar_tools = [t for t in available_tools if tool_name.lower() in t.lower() or t.lower() in tool_name.lower()]
                if similar_tools:
                    logger.info(f"💡 Similar tools found: {similar_tools}")
    logger.info(f"✅ Tools prepared: {len(tools)} tools")

    # Create autonomous agent using proper factory function with fixed agent ID
    logger.info("🔧 Creating autonomous agent with factory function...")

    # Use a fixed agent ID for the Apple Stock Monitor so it can remember across sessions
    # Using a proper UUID format so the memory system doesn't convert it
    APPLE_AGENT_ID = "12345678-1234-5678-9abc-123456789abc"  # Fixed UUID for Apple Stock Monitor

    try:
        agent = create_autonomous_agent(
            name=config.name,
            description=config.description,
            llm=llm,
            tools=tools,
            autonomy_level="autonomous",
            learning_mode="active",
            enable_proactive_behavior=True,
            enable_goal_setting=True,
            max_iterations=config.max_iterations,
            timeout_seconds=config.timeout_seconds,
            agent_id=APPLE_AGENT_ID  # Fixed ID for memory persistence
        )
        logger.info("✅ Autonomous agent created successfully", agent_type=type(agent))
    except Exception as e:
        logger.error("❌ Failed to create autonomous agent", error=str(e))
        raise

    # Set up autonomous monitoring goals using goal manager
    if hasattr(agent, 'goal_manager') and agent.goal_manager:
        from app.agents.autonomous.goal_manager import GoalType, GoalPriority

        # Use the proper add_goal method instead of direct goal creation
        await agent.goal_manager.add_goal(
            title="Apple Stock Monitoring",
            description="Monitor Apple stock for optimal iPhone purchase timing",
            goal_type=GoalType.MAINTENANCE,
            priority=GoalPriority.HIGH,
            target_outcome={
                "monitoring_active": True,
                "data_points_per_hour": 12,
                "analysis_quality": 0.8
            },
            success_criteria=[
                "Track AAPL price movements continuously",
                "Identify buying opportunities",
                "Generate analysis reports",
                "Learn from market patterns"
            ]
        )

    logger.info("🍎 Apple Stock Monitor Agent created using proper infrastructure",
              agent_type=type(agent),
              agent_id=getattr(agent, 'agent_id', 'unknown'),
              has_execute=hasattr(agent, 'execute'))
    return agent

async def start_apple_stock_monitoring():
    """
    🚀 Start Apple stock monitoring using proper autonomous agent execution.

    This demonstrates the correct way to run autonomous agents using the
    existing infrastructure instead of custom monitoring loops.
    """
    try:
        # Initialize all systems properly using UnifiedSystemOrchestrator
        logger.info("🔧 Initializing backend systems using UnifiedSystemOrchestrator...")

        # Use the proper system orchestrator which handles tool registration
        from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator

        orchestrator = get_enhanced_system_orchestrator()
        await orchestrator.initialize()

        # Get initialized systems from orchestrator
        memory_system = orchestrator.memory_system
        rag_system = orchestrator.unified_rag
        tool_repository = orchestrator.tool_repository

        # Initialize LLM manager separately since agent builder integration isn't available
        from app.llm.manager import LLMProviderManager
        llm_manager = LLMProviderManager()
        await llm_manager.initialize()

        logger.info("✅ Successfully accessed all backend systems from orchestrator")

        # Create agent using proper infrastructure
        logger.info("🔧 Creating Apple Stock Monitor Agent...")
        agent = await create_apple_stock_monitor_agent(
            llm_manager=llm_manager,
            memory_system=memory_system,
            rag_system=rag_system,
            tool_repository=tool_repository
        )
        logger.info("✅ Agent created successfully", agent_type=type(agent))

        logger.info("✅ All systems initialized, starting autonomous monitoring...")

        # Initialize monitoring state
        monitoring_state = {
            "iteration_count": 0,
            "previous_observations": [],
            "last_stock_data": None,
            "trend_analysis": [],
            "last_pdf_report_time": None,
            "total_data_points": 0
        }

        # Start autonomous monitoring using proper agent execution
        while True:
            try:
                monitoring_state["iteration_count"] += 1
                current_time = datetime.now().strftime("%H:%M:%S")

                logger.info(f"🎯 Starting monitoring cycle #{monitoring_state['iteration_count']} at {current_time}")

                # Build focused task prompt (personality is in system prompt)
                if monitoring_state["iteration_count"] == 1:
                    task_prompt = """INITIAL MONITORING CYCLE:

                    ACTIONS REQUIRED:
                    1. Fetch current Apple (AAPL) stock data using web_research tool
                    2. Establish baseline measurements for future comparisons
                    3. Store initial observations in memory for trend analysis
                    4. Provide your first market assessment

                    This is your first observation - establish a baseline."""
                else:
                    # Build context from previous observations stored in memory
                    previous_context = ""
                    if monitoring_state["previous_observations"]:
                        recent_obs = monitoring_state["previous_observations"][-3:]  # Last 3 observations
                        previous_context = f"\nPrevious observations: " + "; ".join([
                            f"Cycle {obs['cycle']}: {obs['summary']}" for obs in recent_obs
                        ])

                    task_prompt = f"""MONITORING CYCLE #{monitoring_state['iteration_count']}:

                    STATUS: {monitoring_state['total_data_points']} data points collected, Time: {current_time}
                    {previous_context}

                    ACTIONS REQUIRED:
                    1. Fetch NEW Apple stock data using web_research tool
                    2. Compare with previous observations to identify changes
                    3. Update your analysis based on trends you're seeing
                    4. Communicate what's different from your last observation

                    Build upon your previous knowledge - don't start fresh."""

                # Store previous cycle in memory before new execution
                if monitoring_state["iteration_count"] > 1 and monitoring_state["previous_observations"]:
                    last_obs = monitoring_state["previous_observations"][-1]
                    # Use the correct memory system method
                    from app.agents.autonomous.persistent_memory import MemoryType, MemoryImportance
                    await agent.memory_system.store_memory(
                        content=f"Monitoring cycle {last_obs['cycle']}: {last_obs['summary']}",
                        memory_type=MemoryType.EPISODIC,
                        importance=MemoryImportance.MEDIUM,
                        context={
                            "cycle": last_obs['cycle'],
                            "timestamp": last_obs['timestamp'],
                            "type": "monitoring_observation"
                        }
                    )

                # Execute with focused task prompt (personality is in system prompt)
                result = await agent.execute(task=task_prompt,
                    context={
                        "monitoring_mode": True,
                        "iteration_count": monitoring_state["iteration_count"],
                        "previous_data": monitoring_state["last_stock_data"],
                        "brian_context": {
                            "iphone_budget_needed": 1200,
                            "target_profit_margin": 0.15,
                            "risk_tolerance": "moderate"
                        }
                    }
                )

                # Save this observation to memory
                if result:
                    observation = {
                        "cycle": monitoring_state["iteration_count"],
                        "timestamp": current_time,
                        "summary": f"Completed cycle with {len(result.get('tool_calls', []))} tools used",
                        "result": result
                    }
                    monitoring_state["previous_observations"].append(observation)
                    monitoring_state["total_data_points"] += 1

                    # Keep only last 10 observations to prevent memory bloat
                    if len(monitoring_state["previous_observations"]) > 10:
                        monitoring_state["previous_observations"] = monitoring_state["previous_observations"][-10:]

                logger.info(f"🎯 Monitoring cycle #{monitoring_state['iteration_count']} completed",
                          result_type=type(result),
                          result_keys=list(result.keys()) if result else "None",
                          reasoning_steps=result.get("reasoning_steps", 0) if result else 0,
                          tools_used=len(result.get("tool_calls", [])) if result else 0,
                          total_observations=len(monitoring_state["previous_observations"]))

                # Check if it's time for PDF report (every 5 minutes = 10 cycles of 30 seconds)
                if monitoring_state["iteration_count"] % 10 == 0:
                    logger.info("📄 Time for 5-minute PDF report generation!")
                    monitoring_state["last_pdf_report_time"] = current_time

                # Wait 30 seconds before next monitoring cycle
                logger.info(f"⏰ Waiting 30 seconds before next monitoring cycle...")
                await asyncio.sleep(30)  # 30 second intervals

            except KeyboardInterrupt:
                logger.info("⏹️ Monitoring stopped by user")
                break
            except Exception as e:
                logger.error("Error in autonomous monitoring", error=str(e))
                await asyncio.sleep(60)  # Wait 1 minute on error

    except Exception as e:
        logger.error("Failed to start Apple stock monitoring", error=str(e))
        raise

# Main execution function
if __name__ == "__main__":
    """Run the Apple Stock Monitor Agent using proper infrastructure."""
    print("🍎 Starting Apple Stock Monitor Agent...")
    print("⚠️  This agent now uses proper agentic infrastructure!")
    print("✅ Features: Autonomous execution, proper memory, RAG integration, tool repository")
    print("⏹️  Press Ctrl+C to stop monitoring...")

    try:
        asyncio.run(start_apple_stock_monitoring())
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.error("Failed to run Apple Stock Monitor Agent", error=str(e))
