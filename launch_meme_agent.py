#!/usr/bin/env python3
"""
🎭 AUTONOMOUS MEME AGENT LAUNCHER
Launch your revolutionary autonomous meme-generating agent!

This script starts your fully autonomous meme agent that will:
- Continuously collect memes from Reddit, Imgur, 9GAG
- Learn patterns from collected memes using AI analysis
- Generate new original memes based on learned patterns
- Save all memes to the 'memes' directory
- Run 24/7 autonomously with memory and learning capabilities

Usage:
    python launch_meme_agent.py

The agent will create and manage:
- data/memes/collected/     - Downloaded memes from the web
- data/memes/generated/     - AI-generated original memes
- data/memes/analysis_cache/ - Analysis data for learning
- data/memes/templates/     - Meme templates for generation
"""

import asyncio
import sys
import os
import signal
from pathlib import Path
from typing import Optional

import structlog

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Import the autonomous meme agent and supporting systems
from app.agents.autonomous.meme_agent import AutonomousMemeAgent, MemeAgentConfig, create_autonomous_meme_agent
from app.memory.unified_memory_system import UnifiedMemorySystem
from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.llm.manager import get_enhanced_llm_manager
from app.llm.models import LLMConfig, ProviderType
from app.config.settings import get_settings

# Setup logging
logger = structlog.get_logger(__name__)


class MemeAgentLauncher:
    """Launcher for the autonomous meme agent."""
    
    def __init__(self):
        self.agent: Optional[AutonomousMemeAgent] = None
        self.llm_manager = None
        self.memory_system = None
        self.rag_system = None
        self.running = False
        
    async def initialize_systems(self) -> bool:
        """Initialize all required systems."""
        try:
            print("🔧 Initializing LLM Manager...")
            self.llm_manager = get_enhanced_llm_manager()
            settings = get_settings()
            credentials = settings.get_provider_credentials()
            await self.llm_manager.initialize(credentials)
            
            available_providers = self.llm_manager.get_available_providers()
            print(f"  ✅ LLM Manager initialized with providers: {available_providers}")
            
            if not available_providers:
                print("  ❌ No LLM providers available. Please check your configuration.")
                return False
            
            print("🧠 Initializing Memory System...")
            self.memory_system = UnifiedMemorySystem()
            await self.memory_system.initialize()
            print("  ✅ Unified Memory System initialized")
            
            print("📚 Initializing RAG System...")
            self.rag_system = UnifiedRAGSystem()
            await self.rag_system.initialize()
            print("  ✅ Unified RAG System initialized")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize systems: {str(e)}")
            return False
    
    async def create_meme_agent(self) -> bool:
        """Create and configure the autonomous meme agent."""
        try:
            print("🎭 Creating Autonomous Meme Agent...")
            
            # Get LLM for the agent
            llm_config = LLMConfig(
                provider=ProviderType.OLLAMA,  # Default to Ollama
                model_id="llama3.2:latest",   # Use available model
                temperature=0.8               # Creative temperature for memes
            )
            llm = await self.llm_manager.get_llm(llm_config)
            
            # Configure the meme agent
            meme_config = MemeAgentConfig(
                agent_id="autonomous_meme_agent_001",
                collection_interval_hours=2,    # Collect every 2 hours
                generation_interval_hours=4,    # Generate every 4 hours
                max_memes_per_collection=50,    # Collect up to 50 memes per run
                min_quality_threshold=0.6,      # Only keep quality memes
                storage_directory="memes",      # Save to memes directory
                enable_proactive_generation=True,
                enable_trend_following=True,
                target_subreddits=[
                    'memes', 'dankmemes', 'wholesomememes', 
                    'PrequelMemes', 'HistoryMemes', 'ProgrammerHumor', 
                    'me_irl', 'funny', 'AdviceAnimals'
                ]
            )
            
            # Create the autonomous meme agent
            self.agent = AutonomousMemeAgent(
                llm=llm,
                config=meme_config,
                unified_memory=self.memory_system,
                unified_rag=self.rag_system
            )
            
            print("  ✅ Autonomous Meme Agent created successfully!")
            print(f"  📁 Memes will be saved to: {meme_config.storage_directory}/")
            print(f"  🎯 Target subreddits: {', '.join(meme_config.target_subreddits[:5])}...")
            print(f"  ⏰ Collection interval: {meme_config.collection_interval_hours} hours")
            print(f"  🎨 Generation interval: {meme_config.generation_interval_hours} hours")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create meme agent: {str(e)}")
            return False
    
    async def start_agent(self):
        """Start the autonomous meme agent operation."""
        try:
            print("\n🚀 Starting Autonomous Meme Agent Operation...")
            print("=" * 60)
            
            # Ensure memes directory exists
            memes_dir = Path("memes")
            memes_dir.mkdir(exist_ok=True)
            (memes_dir / "collected").mkdir(exist_ok=True)
            (memes_dir / "generated").mkdir(exist_ok=True)
            (memes_dir / "analysis_cache").mkdir(exist_ok=True)
            (memes_dir / "templates").mkdir(exist_ok=True)
            
            print(f"📁 Memes directory structure created at: {memes_dir.absolute()}")
            
            # Start the autonomous operation
            self.running = True
            await self.agent.start_autonomous_operation()
            
        except Exception as e:
            logger.error(f"Failed to start meme agent: {str(e)}")
            self.running = False
    
    async def run(self):
        """Main run method."""
        print("🎭 AUTONOMOUS MEME AGENT LAUNCHER")
        print("=" * 60)
        print("🤖 Your revolutionary autonomous meme-generating agent!")
        print("🌐 Collects memes from Reddit, Imgur, 9GAG")
        print("🧠 Learns patterns and generates original memes")
        print("💾 Saves everything to the 'memes' directory")
        print("🔄 Runs continuously 24/7")
        print("=" * 60)
        
        # Initialize all systems
        if not await self.initialize_systems():
            print("❌ System initialization failed!")
            return False
        
        # Create the meme agent
        if not await self.create_meme_agent():
            print("❌ Meme agent creation failed!")
            return False
        
        # Start the agent
        await self.start_agent()
        
        return True
    
    def shutdown(self):
        """Graceful shutdown."""
        print("\n🛑 Shutting down Autonomous Meme Agent...")
        self.running = False
        if self.agent:
            # The agent will handle its own cleanup
            pass
        print("👋 Meme agent stopped. Generated memes saved in 'memes' directory!")


async def main():
    """Main entry point."""
    launcher = MemeAgentLauncher()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        launcher.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        success = await launcher.run()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        launcher.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    print("🎭 Starting Autonomous Meme Agent...")
    asyncio.run(main())
