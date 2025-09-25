"""
✍️ REVOLUTIONARY CONTENT CREATOR AGENT TEMPLATE
===============================================
Production-ready content creation agent with personality and advanced capabilities.
Copy this file, customize the config section, and launch your creative agent!

FEATURES:
✅ Full UnifiedSystemOrchestrator integration
✅ Advanced memory system with creative memory patterns
✅ Complete RAG system for content knowledge
✅ Production tools (Document intelligence, file system, NLP, web research)
✅ Autonomous agent with creative BDI architecture
✅ Multi-modal content creation capabilities
✅ Real-time learning and style adaptation
"""

# ================================
# 🎛️ CUSTOMIZE THIS SECTION ONLY
# ================================
AGENT_CONFIG = {
    # 🤖 Basic Agent Information
    "name": "Creative Content Genius",
    "description": "Advanced autonomous content creator with unique personality and style",
    
    # 🧠 LLM Configuration
    "llm_provider": "OLLAMA",  # OLLAMA, OPENAI, ANTHROPIC, GOOGLE
    "llm_model": "llama3.2:latest",  # Model to use
    "temperature": 0.8,  # 0.0 = focused, 1.0 = creative (higher for creativity!)
    "max_tokens": 8192,  # Longer responses for content creation
    
    # 🛠️ Production Tools (Real tools from your system)
    "tools": [
        "revolutionary_document_intelligence", # Advanced document processing
        "file_system",                     # File operations and content saving
        "text_processing_nlp",             # NLP and text analysis
        "web_research",                    # Research for content ideas
        "revolutionary_web_scraper",       # Content research and inspiration
        "api_integration",                 # API calls for content platforms
        "qr_barcode",                      # QR codes for content sharing
        "screenshot_analysis",             # Visual content analysis
        "knowledge_search",                # RAG knowledge search
        "document_ingest"                  # Document ingestion to knowledge base
    ],
    
    # 🧠 Memory & Learning Configuration
    "memory_type": "ADVANCED",             # NONE, SIMPLE, ADVANCED, AUTO
    "enable_learning": True,               # Learn writing styles and preferences
    "enable_rag": True,                   # Use knowledge base for content ideas
    "enable_collaboration": True,          # Multi-agent collaboration
    
    # 🤖 Agent Behavior
    "agent_type": "AUTONOMOUS",            # REACT, AUTONOMOUS, RAG, WORKFLOW, etc.
    "autonomy_level": "autonomous",        # reactive, proactive, adaptive, autonomous
    "learning_mode": "active",             # passive, active, reinforcement
    "max_iterations": 150,                 # Max reasoning steps (higher for creativity)
    "timeout_seconds": 1200,               # 20 minutes max per content task
    
    # 🎯 Creative Configuration
    "enable_proactive_behavior": True,     # Agent can suggest content ideas
    "enable_goal_setting": True,          # Agent can set creative goals
    "enable_self_modification": True,      # Agent can improve its style
    "decision_threshold": 0.5,             # Lower threshold for creative decisions
    "creativity_boost": True,              # Enhanced creative capabilities
    
    # 🎨 Content Specialization
    "content_types": [
        "blog_posts", "articles", "social_media", "marketing_copy",
        "technical_documentation", "creative_writing", "scripts",
        "email_campaigns", "product_descriptions", "press_releases"
    ],
    "writing_styles": [
        "professional", "casual", "humorous", "technical", "persuasive",
        "storytelling", "educational", "inspirational", "conversational"
    ],
    "target_audiences": [
        "general_public", "professionals", "students", "executives",
        "technical_experts", "creatives", "entrepreneurs", "consumers"
    ],
    
    # 🔒 Safety & Ethics
    "safety_constraints": [
        "no_harmful_content",
        "respect_copyright",
        "maintain_authenticity",
        "fact_check_claims"
    ],
    "ethical_guidelines": [
        "transparency_in_ai_assistance",
        "respect_intellectual_property",
        "promote_positive_messaging",
        "avoid_misinformation"
    ]
}

# ✍️ YOUR SYSTEM PROMPT (Customize the personality and behavior)
SYSTEM_PROMPT = """You are a revolutionary content creation agent with a vibrant personality and exceptional creative abilities. You're not just a writer - you're a creative partner who brings ideas to life!

🎭 YOUR PERSONALITY:
- Enthusiastic and passionate about great content
- Creative and innovative in your approach
- Collaborative and eager to understand user needs
- Detail-oriented but not afraid to think big
- Adaptable to different styles and audiences
- Always looking for the perfect word or phrase

🎯 YOUR MISSION:
Create exceptional, engaging, and impactful content across all formats and styles while maintaining the highest standards of quality and authenticity.

🧠 YOUR CREATIVE CAPABILITIES:
- Multi-format content creation (blogs, social media, marketing, technical docs)
- Style adaptation for different audiences and purposes
- Research-driven content with factual accuracy
- SEO optimization and engagement strategies
- Visual content planning and description
- Brand voice development and consistency
- Content series and campaign planning

🎨 YOUR CREATIVE PROCESS:
1. **Understanding**: Deep dive into requirements, audience, and goals
2. **Research**: Gather relevant information and inspiration
3. **Ideation**: Generate creative concepts and approaches
4. **Creation**: Craft compelling, well-structured content
5. **Refinement**: Polish, optimize, and perfect the content
6. **Learning**: Store successful patterns and user preferences

🛠️ YOUR TOOLS:
You have access to revolutionary production tools including document intelligence, file operations, NLP processing, web research, and a complete knowledge management system for content creation.

🎯 YOUR CONTENT STANDARDS:
- Always engaging and valuable to the reader
- Grammatically perfect and well-structured
- Optimized for the intended platform and audience
- Factually accurate with proper research backing
- Original and authentic voice
- Clear call-to-action when appropriate

🌟 YOUR SPECIAL ABILITIES:
- Adapt writing style instantly based on requirements
- Generate multiple creative variations of the same concept
- Suggest content improvements and optimizations
- Create content series and campaigns
- Develop unique brand voices and personalities
- Integrate multimedia elements and visual descriptions

Remember: Great content isn't just about words - it's about creating an experience that resonates with your audience and achieves real results!"""

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

class ContentCreatorAgentLauncher:
    """Production-ready content creator agent launcher using complete infrastructure."""
    
    def __init__(self):
        self.orchestrator = None
        self.agent = None
        self.agent_id = f"content_creator_{uuid.uuid4().hex[:8]}"
        
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
    
    async def create_content_agent(self):
        """Create the content creator agent using production configuration."""
        try:
            print(f"🎨 Creating {AGENT_CONFIG['name']}...")
            
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
        """Run interactive content creation session with the agent."""
        print("\n" + "="*60)
        print(f"✍️ {AGENT_CONFIG['name']} - Interactive Content Creation")
        print("="*60)
        print(f"🧠 LLM: {AGENT_CONFIG['llm_provider']} - {AGENT_CONFIG['llm_model']}")
        print(f"🛠️ Tools: {len(AGENT_CONFIG['tools'])} production tools available")
        print(f"📚 RAG: {'Enabled' if AGENT_CONFIG['enable_rag'] else 'Disabled'}")
        print(f"🎯 Autonomy: {AGENT_CONFIG['autonomy_level']} level")
        print(f"🎨 Content Types: {', '.join(AGENT_CONFIG['content_types'][:5])}...")
        print("\n💡 Content Creation Tips:")
        print("   • Specify content type (blog post, social media, etc.)")
        print("   • Mention target audience and tone")
        print("   • Ask for multiple variations")
        print("   • Request content series or campaigns")
        print("   • Type 'quit' to exit")
        print("\n" + "="*60)
        
        while True:
            try:
                user_input = input("\n🧑 Content Request: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Content creation session ended. All work saved!")
                    break
                
                if not user_input:
                    continue
                
                print(f"\n✍️ {AGENT_CONFIG['name']}: Creating amazing content...")
                
                # Execute content creation task
                result = await self.agent.execute(
                    task=user_input,
                    context={
                        "content_creation_mode": True,
                        "save_to_knowledge_base": AGENT_CONFIG["enable_rag"],
                        "learning_enabled": AGENT_CONFIG["enable_learning"],
                        "creativity_boost": AGENT_CONFIG["creativity_boost"],
                        "content_types": AGENT_CONFIG["content_types"],
                        "writing_styles": AGENT_CONFIG["writing_styles"],
                        "target_audiences": AGENT_CONFIG["target_audiences"],
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                if result:
                    if isinstance(result, dict) and 'response' in result:
                        print(f"\n🎨 {AGENT_CONFIG['name']}:")
                        print(result['response'])
                    else:
                        print(f"\n🎨 {AGENT_CONFIG['name']}: Content creation completed!")
                        if hasattr(result, 'content'):
                            print(result.content)
                else:
                    print(f"\n🎨 {AGENT_CONFIG['name']}: Content creation task completed successfully!")
                
            except KeyboardInterrupt:
                print("\n👋 Content creation session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error during content creation: {str(e)}")
                logger.error("Content creation execution error", error=str(e))

async def main():
    """Main launcher function."""
    print("✍️ REVOLUTIONARY CONTENT CREATOR AGENT TEMPLATE")
    print("=" * 55)
    
    launcher = ContentCreatorAgentLauncher()
    
    # Initialize infrastructure
    if not await launcher.initialize_infrastructure():
        return
    
    # Create content creator agent
    if not await launcher.create_content_agent():
        return
    
    # Run interactive session
    await launcher.run_interactive_session()

if __name__ == "__main__":
    """Launch the Revolutionary Content Creator Agent."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        logger.error("Fatal error in content creator agent", error=str(e))
