"""
🔬 REVOLUTIONARY RESEARCH AGENT TEMPLATE
========================================
Production-ready research agent using the complete Agentic AI infrastructure.
Copy this file, customize the config section, and launch your research agent!

FEATURES:
✅ Full UnifiedSystemOrchestrator integration
✅ Advanced memory system (UnifiedMemorySystem + PersistentMemorySystem)
✅ Complete RAG system (UnifiedRAGSystem + ChromaDB)
✅ Production tools (Web research, document intelligence, file system)
✅ Autonomous agent with BDI architecture
✅ Real-time learning and adaptation
✅ Multi-modal capabilities
"""

# ================================
# 🎛️ CUSTOMIZE THIS SECTION ONLY
# ================================
AGENT_CONFIG = {
    # 🤖 Basic Agent Information
    "name": "My Research Assistant",
    "description": "Advanced autonomous research agent with deep analysis capabilities",
    
    # 🧠 LLM Configuration
    "llm_provider": "OLLAMA",  # OLLAMA, OPENAI, ANTHROPIC, GOOGLE
    "llm_model": "llama3.2:latest",  # Model to use
    "temperature": 0.3,  # 0.0 = focused, 1.0 = creative
    "max_tokens": 4096,  # Response length
    
    # 🛠️ Production Tools (Real tools from your system)
    "tools": [
        "web_research",                    # Revolutionary web research with AI
        "revolutionary_web_scraper",       # Ultimate web scraping system
        "revolutionary_document_intelligence", # Advanced document processing
        "file_system",                     # File operations
        "text_processing_nlp",             # NLP and text analysis
        "api_integration",                 # API calls and integrations
        "calculator",                      # Mathematical calculations
        "knowledge_search",                # RAG knowledge search
        "document_ingest"                  # Document ingestion to knowledge base
    ],
    
    # 🧠 Memory & Learning Configuration
    "memory_type": "ADVANCED",             # NONE, SIMPLE, ADVANCED, AUTO
    "enable_learning": True,               # Learn from interactions
    "enable_rag": True,                   # Use knowledge base
    "enable_collaboration": True,          # Multi-agent collaboration
    
    # 🤖 Agent Behavior
    "agent_type": "AUTONOMOUS",            # REACT, AUTONOMOUS, RAG, WORKFLOW, etc.
    "autonomy_level": "autonomous",        # reactive, proactive, adaptive, autonomous
    "learning_mode": "active",             # passive, active, reinforcement
    "max_iterations": 100,                 # Max reasoning steps
    "timeout_seconds": 900,                # 15 minutes max per task
    
    # 🎯 Specialized Configuration
    "enable_proactive_behavior": True,     # Agent can initiate actions
    "enable_goal_setting": True,          # Agent can set its own goals
    "enable_self_modification": True,      # Agent can improve itself
    "decision_threshold": 0.6,             # Decision confidence threshold
    
    # 🔒 Safety & Ethics
    "safety_constraints": [
        "verify_information_sources",
        "maintain_research_ethics", 
        "respect_intellectual_property",
        "no_harmful_content"
    ],
    "ethical_guidelines": [
        "transparency_in_research",
        "cite_all_sources",
        "acknowledge_limitations"
    ]
}

# ✍️ YOUR SYSTEM PROMPT (Customize the personality and behavior)
SYSTEM_PROMPT = """You are an advanced autonomous research assistant with deep analytical capabilities and a methodical approach to investigation.

🎯 YOUR MISSION:
Conduct comprehensive, accurate, and insightful research on any topic using your advanced tools and reasoning capabilities.

🧠 YOUR CAPABILITIES:
- Advanced web research with AI-powered analysis
- Document intelligence and processing
- Knowledge base integration and management
- Multi-source information synthesis
- Real-time learning and adaptation
- Autonomous goal setting and planning

🔬 YOUR RESEARCH METHODOLOGY:
1. **Topic Analysis**: Break down complex topics into research questions
2. **Multi-Source Investigation**: Use web research, documents, and knowledge base
3. **Information Verification**: Cross-reference sources and verify facts
4. **Synthesis & Analysis**: Combine findings into coherent insights
5. **Knowledge Storage**: Store important findings in your knowledge base
6. **Continuous Learning**: Adapt your approach based on results

🎭 YOUR PERSONALITY:
- Methodical and systematic in approach
- Curious about details and nuances
- Intellectually honest about limitations
- Proactive in suggesting related research
- Clear and engaging in communication
- Always cite sources and provide evidence

🛠️ YOUR TOOLS:
You have access to revolutionary production tools including web research, document intelligence, file operations, NLP processing, and a complete knowledge management system.

🎯 YOUR GOALS:
- Provide accurate, comprehensive research
- Build and maintain a knowledge base
- Learn from each research task
- Suggest follow-up investigations
- Maintain the highest research standards

Always strive for accuracy, depth, and actionable insights while maintaining research ethics and transparency!"""

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

class ResearchAgentLauncher:
    """Production-ready research agent launcher using complete infrastructure."""
    
    def __init__(self):
        self.orchestrator = None
        self.agent = None
        self.agent_id = f"research_agent_{uuid.uuid4().hex[:8]}"
        
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
    
    async def create_research_agent(self):
        """Create the research agent using production configuration."""
        try:
            print(f"🤖 Creating {AGENT_CONFIG['name']}...")
            
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
                safety_constraints=AGENT_CONFIG["safety_constraints"],
                ethical_guidelines=AGENT_CONFIG["ethical_guidelines"],
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
        """Run interactive research session with the agent."""
        print("\n" + "="*60)
        print(f"🔬 {AGENT_CONFIG['name']} - Interactive Research Session")
        print("="*60)
        print(f"🧠 LLM: {AGENT_CONFIG['llm_provider']} - {AGENT_CONFIG['llm_model']}")
        print(f"🛠️ Tools: {len(AGENT_CONFIG['tools'])} production tools available")
        print(f"📚 RAG: {'Enabled' if AGENT_CONFIG['enable_rag'] else 'Disabled'}")
        print(f"🎯 Autonomy: {AGENT_CONFIG['autonomy_level']} level")
        print("\n💡 Tips:")
        print("   • Ask for research on any topic")
        print("   • Request document analysis")
        print("   • Ask to save findings to knowledge base")
        print("   • Type 'quit' to exit")
        print("\n" + "="*60)
        
        while True:
            try:
                user_input = input("\n🧑 Research Request: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Research session ended. Knowledge saved!")
                    break
                
                if not user_input:
                    continue
                
                print(f"\n🔬 {AGENT_CONFIG['name']}: Conducting research...")
                
                # Execute research task
                result = await self.agent.execute(
                    task=user_input,
                    context={
                        "research_mode": True,
                        "save_to_knowledge_base": AGENT_CONFIG["enable_rag"],
                        "learning_enabled": AGENT_CONFIG["enable_learning"],
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                if result:
                    if isinstance(result, dict) and 'response' in result:
                        print(f"\n🤖 {AGENT_CONFIG['name']}:")
                        print(result['response'])
                    else:
                        print(f"\n🤖 {AGENT_CONFIG['name']}: Research completed!")
                        if hasattr(result, 'content'):
                            print(result.content)
                else:
                    print(f"\n🤖 {AGENT_CONFIG['name']}: Research task completed successfully!")
                
            except KeyboardInterrupt:
                print("\n👋 Research session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error during research: {str(e)}")
                logger.error("Research execution error", error=str(e))

async def main():
    """Main launcher function."""
    print("🔬 REVOLUTIONARY RESEARCH AGENT TEMPLATE")
    print("=" * 50)
    
    launcher = ResearchAgentLauncher()
    
    # Initialize infrastructure
    if not await launcher.initialize_infrastructure():
        return
    
    # Create research agent
    if not await launcher.create_research_agent():
        return
    
    # Run interactive session
    await launcher.run_interactive_session()

if __name__ == "__main__":
    """Launch the Revolutionary Research Agent."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        logger.error("Fatal error in research agent", error=str(e))
