"""
📊 REVOLUTIONARY BUSINESS INTELLIGENCE AGENT TEMPLATE
====================================================
Production-ready business intelligence agent with data analysis and reporting capabilities.
Copy this file, customize the config section, and launch your BI agent!

FEATURES:
✅ Full UnifiedSystemOrchestrator integration
✅ Advanced memory system with analytical patterns
✅ Complete RAG system for business knowledge
✅ Production BI tools (Database operations, text processing, web research)
✅ Autonomous agent with analytical BDI architecture
✅ Advanced data processing and visualization planning
✅ Real-time learning and insight generation
"""

# ================================
# 🎛️ CUSTOMIZE THIS SECTION ONLY
# ================================
AGENT_CONFIG = {
    # 🤖 Basic Agent Information
    "name": "Business Intelligence Master",
    "description": "Advanced autonomous business intelligence agent with data analysis and reporting capabilities",
    
    # 🧠 LLM Configuration
    "llm_provider": "OLLAMA",  # OLLAMA, OPENAI, ANTHROPIC, GOOGLE
    "llm_model": "llama3.2:latest",  # Model to use
    "temperature": 0.1,  # 0.0 = focused, 1.0 = creative (very low for accuracy)
    "max_tokens": 8192,  # Longer responses for detailed analysis
    
    # 🛠️ Production Tools (Real BI tools from your system)
    "tools": [
        "database_operations",             # Database queries and management
        "text_processing_nlp",             # Text analysis and NLP
        "web_research",                    # Market research and data gathering
        "revolutionary_web_scraper",       # Data extraction and collection
        "api_integration",                 # API calls for data sources
        "file_system",                     # File operations and data management
        "revolutionary_document_intelligence", # Document analysis and processing
        "calculator",                      # Mathematical calculations
        "business_intelligence",           # Specialized BI tool
        "knowledge_search",                # RAG knowledge search
        "document_ingest"                  # Document ingestion to knowledge base
    ],
    
    # 🧠 Memory & Learning Configuration
    "memory_type": "ADVANCED",             # NONE, SIMPLE, ADVANCED, AUTO
    "enable_learning": True,               # Learn analytical patterns
    "enable_rag": True,                   # Use knowledge base for business insights
    "enable_collaboration": True,          # Multi-agent collaboration
    
    # 🤖 Agent Behavior
    "agent_type": "AUTONOMOUS",            # REACT, AUTONOMOUS, RAG, WORKFLOW, etc.
    "autonomy_level": "autonomous",        # reactive, proactive, adaptive, autonomous
    "learning_mode": "active",             # passive, active, reinforcement
    "max_iterations": 150,                 # Max reasoning steps for complex analysis
    "timeout_seconds": 1800,               # 30 minutes max per analysis task
    
    # 🎯 BI Configuration
    "enable_proactive_behavior": True,     # Agent can suggest analyses
    "enable_goal_setting": True,          # Agent can set analytical goals
    "enable_self_modification": True,      # Agent can improve analysis methods
    "decision_threshold": 0.8,             # High threshold for business decisions
    "analytical_depth": "comprehensive",   # surface, standard, comprehensive, deep
    
    # 📊 Business Intelligence Specialization
    "analysis_types": [
        "financial_analysis", "market_research", "customer_analytics",
        "operational_metrics", "competitive_intelligence", "trend_analysis",
        "performance_dashboards", "predictive_analytics", "risk_assessment"
    ],
    "data_sources": [
        "databases", "apis", "web_scraping", "documents", "spreadsheets",
        "social_media", "market_feeds", "internal_systems", "external_reports"
    ],
    "reporting_formats": [
        "executive_summary", "detailed_report", "dashboard_design",
        "presentation_slides", "data_visualization", "kpi_tracking",
        "trend_analysis", "comparative_analysis", "forecasting_report"
    ],
    
    # 🔒 Data Security & Compliance
    "security_constraints": [
        "data_privacy_protection",
        "secure_data_handling",
        "audit_trail_maintenance",
        "access_control_compliance"
    ],
    "compliance_guidelines": [
        "gdpr_compliance",
        "financial_regulations",
        "industry_standards",
        "data_governance_policies"
    ]
}

# ✍️ YOUR SYSTEM PROMPT (Customize the personality and behavior)
SYSTEM_PROMPT = """You are an advanced autonomous business intelligence agent with exceptional analytical capabilities and deep business acumen. You're the ultimate data-driven decision support system!

🎯 YOUR MISSION:
Transform raw data into actionable business insights, create comprehensive analyses, and provide strategic recommendations that drive business success.

📊 YOUR ANALYTICAL CAPABILITIES:
- Advanced data analysis and statistical modeling
- Market research and competitive intelligence
- Financial analysis and performance metrics
- Customer analytics and segmentation
- Operational efficiency analysis
- Trend identification and forecasting
- Risk assessment and mitigation strategies
- KPI development and tracking

🔍 YOUR ANALYTICAL METHODOLOGY:
1. **Requirements Gathering**: Understand business questions and objectives
2. **Data Discovery**: Identify and access relevant data sources
3. **Data Analysis**: Apply statistical and analytical techniques
4. **Pattern Recognition**: Identify trends, correlations, and insights
5. **Insight Generation**: Develop actionable business recommendations
6. **Visualization Planning**: Design effective data presentations
7. **Report Creation**: Produce comprehensive analytical reports
8. **Knowledge Storage**: Store insights for future reference

🎭 YOUR PERSONALITY:
- Analytical and detail-oriented
- Strategic and business-focused
- Data-driven and evidence-based
- Proactive in identifying opportunities
- Clear and concise in communication
- Ethical and compliant in data handling

🛠️ YOUR TOOLS:
You have access to revolutionary BI tools including database operations, advanced analytics, web research, document intelligence, and a complete business knowledge management system.

📈 YOUR ANALYTICAL STANDARDS:
- Accurate and reliable data analysis
- Statistically sound methodologies
- Clear and actionable insights
- Comprehensive documentation
- Ethical data handling practices
- Business-relevant recommendations

🌟 YOUR SPECIAL ABILITIES:
- Multi-source data integration and analysis
- Advanced pattern recognition and trend analysis
- Strategic business insight generation
- Automated report and dashboard creation
- Predictive analytics and forecasting
- Competitive intelligence gathering

🔒 YOUR DATA GOVERNANCE:
- Maintain strict data privacy and security
- Ensure compliance with regulations
- Create comprehensive audit trails
- Respect access controls and permissions
- Handle sensitive information appropriately
- Follow industry best practices

Remember: Great business intelligence is not just about analyzing data - it's about transforming information into strategic advantages that drive business growth and success!"""

# ================================
# 🚀 LAUNCH CODE (DON'T TOUCH!)
# ================================
import asyncio
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
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

class BusinessIntelligenceAgentLauncher:
    """Production-ready business intelligence agent launcher using complete infrastructure."""
    
    def __init__(self):
        self.orchestrator = None
        self.agent = None
        self.agent_id = f"bi_agent_{uuid.uuid4().hex[:8]}"
        
    async def initialize_infrastructure(self):
        """Initialize the complete production infrastructure."""
        try:
            print("🔧 Initializing Revolutionary Agentic AI Infrastructure...")
            
            # Get the enhanced system orchestrator (THE unified system)
            self.orchestrator = get_enhanced_system_orchestrator()
            await self.orchestrator.initialize()
            
            print("✅ Infrastructure initialized successfully!")
            print(f"   🧠 Memory System: {type(self.orchestrator.memory_system).__name__}")
            print(f"   📚 RAG System: {type(self.orchestrator.unified_rag).__name__}")
            print(f"   🛠️ Tool Repository: {self.orchestrator.tool_repository.stats['total_tools']} tools available")
            
            return True
            
        except Exception as e:
            print(f"❌ Infrastructure initialization failed: {str(e)}")
            logger.error("Infrastructure initialization failed", error=str(e))
            return False
    
    async def create_bi_agent(self):
        """Create the business intelligence agent using production configuration."""
        try:
            print(f"📊 Creating {AGENT_CONFIG['name']}...")
            
            # Create LLM configuration
            llm_config = LLMConfig(
                provider=ProviderType(AGENT_CONFIG["llm_provider"]),
                model_id=AGENT_CONFIG["llm_model"],
                temperature=AGENT_CONFIG["temperature"],
                max_tokens=AGENT_CONFIG["max_tokens"]
            )
            
            # Initialize LLM manager and create LLM instance
            llm_manager = LLMProviderManager()
            await llm_manager.initialize()
            llm = await llm_manager.create_llm_instance(llm_config)
            
            # Get tools from the unified tool repository
            tools = []
            for tool_name in AGENT_CONFIG["tools"]:
                tool = self.orchestrator.tool_repository.get_tool(tool_name)
                if tool:
                    tools.append(tool)
                    print(f"   ✅ Tool loaded: {tool_name}")
                else:
                    print(f"   ⚠️ Tool not found: {tool_name}")
            
            print(f"   🛠️ {len(tools)} tools loaded successfully")
            
            # Create autonomous agent using the factory function
            self.agent = create_autonomous_agent(
                name=AGENT_CONFIG["name"],
                description=AGENT_CONFIG["description"],
                llm=llm,
                tools=tools,
                autonomy_level=AGENT_CONFIG["autonomy_level"],
                learning_mode=AGENT_CONFIG["learning_mode"],
                agent_id=self.agent_id,
                enable_proactive_behavior=AGENT_CONFIG["enable_proactive_behavior"],
                enable_goal_setting=AGENT_CONFIG["enable_goal_setting"],
                enable_self_modification=AGENT_CONFIG["enable_self_modification"],
                decision_threshold=AGENT_CONFIG["decision_threshold"],
                safety_constraints=AGENT_CONFIG["security_constraints"],
                ethical_guidelines=AGENT_CONFIG["compliance_guidelines"],
                max_iterations=AGENT_CONFIG["max_iterations"],
                timeout_seconds=AGENT_CONFIG["timeout_seconds"]
            )
            
            # Set the system prompt
            self.agent.system_prompt = SYSTEM_PROMPT
            
            # Initialize agent memory and RAG
            if AGENT_CONFIG["enable_rag"]:
                await self.orchestrator.unified_rag.create_agent_knowledge_base(self.agent_id)
                print("   📚 Knowledge base created")
            
            await self.orchestrator.memory_system.create_agent_memory(self.agent_id)
            print("   🧠 Memory system initialized")
            
            # Create tool profile for the agent
            await self.orchestrator.tool_repository.create_agent_profile(self.agent_id)
            for tool_name in AGENT_CONFIG["tools"]:
                await self.orchestrator.tool_repository.assign_tool_to_agent(self.agent_id, tool_name)
            print("   🛠️ Tool profile created")
            
            print(f"✅ {AGENT_CONFIG['name']} created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Agent creation failed: {str(e)}")
            logger.error("Agent creation failed", error=str(e))
            return False
    
    async def run_interactive_session(self):
        """Run interactive business intelligence session with the agent."""
        print("\n" + "="*60)
        print(f"📊 {AGENT_CONFIG['name']} - Interactive Business Intelligence")
        print("="*60)
        print(f"🧠 LLM: {AGENT_CONFIG['llm_provider']} - {AGENT_CONFIG['llm_model']}")
        print(f"🛠️ Tools: {len(AGENT_CONFIG['tools'])} BI tools available")
        print(f"📚 RAG: {'Enabled' if AGENT_CONFIG['enable_rag'] else 'Disabled'}")
        print(f"🎯 Autonomy: {AGENT_CONFIG['autonomy_level']} level")
        print(f"📈 Analysis Types: {', '.join(AGENT_CONFIG['analysis_types'][:4])}...")
        print("\n💡 Business Intelligence Tips:")
        print("   • Ask for market analysis or competitive intelligence")
        print("   • Request financial performance analysis")
        print("   • Ask for customer analytics and segmentation")
        print("   • Request trend analysis and forecasting")
        print("   • Ask for KPI development and tracking")
        print("   • Type 'quit' to exit")
        print("\n🔒 Data Security: All analyses follow strict compliance guidelines")
        print("\n" + "="*60)
        
        while True:
            try:
                user_input = input("\n🧑 BI Analysis Request: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Business intelligence session ended. All insights saved!")
                    break
                
                if not user_input:
                    continue
                
                print(f"\n📊 {AGENT_CONFIG['name']}: Conducting business analysis...")
                
                # Execute BI analysis task
                result = await self.agent.execute(
                    task=user_input,
                    context={
                        "business_intelligence_mode": True,
                        "save_to_knowledge_base": AGENT_CONFIG["enable_rag"],
                        "learning_enabled": AGENT_CONFIG["enable_learning"],
                        "analytical_depth": AGENT_CONFIG["analytical_depth"],
                        "analysis_types": AGENT_CONFIG["analysis_types"],
                        "data_sources": AGENT_CONFIG["data_sources"],
                        "reporting_formats": AGENT_CONFIG["reporting_formats"],
                        "security_constraints": AGENT_CONFIG["security_constraints"],
                        "compliance_guidelines": AGENT_CONFIG["compliance_guidelines"],
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                if result:
                    if isinstance(result, dict) and 'response' in result:
                        print(f"\n📈 {AGENT_CONFIG['name']}:")
                        print(result['response'])
                    else:
                        print(f"\n📈 {AGENT_CONFIG['name']}: Business analysis completed!")
                        if hasattr(result, 'content'):
                            print(result.content)
                else:
                    print(f"\n📈 {AGENT_CONFIG['name']}: Business intelligence task completed successfully!")
                
            except KeyboardInterrupt:
                print("\n👋 BI session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error during analysis: {str(e)}")
                logger.error("BI analysis execution error", error=str(e))

async def main():
    """Main launcher function."""
    print("📊 REVOLUTIONARY BUSINESS INTELLIGENCE AGENT TEMPLATE")
    print("=" * 60)
    
    launcher = BusinessIntelligenceAgentLauncher()
    
    # Initialize infrastructure
    if not await launcher.initialize_infrastructure():
        return
    
    # Create BI agent
    if not await launcher.create_bi_agent():
        return
    
    # Run interactive session
    await launcher.run_interactive_session()

if __name__ == "__main__":
    """Launch the Revolutionary Business Intelligence Agent."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        logger.error("Fatal error in BI agent", error=str(e))
