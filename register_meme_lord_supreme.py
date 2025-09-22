#!/usr/bin/env python3
"""
🎭👑 MEME LORD SUPREME AGENT REGISTRATION SCRIPT 👑🎭

This script registers the Meme Lord Supreme Agent with your existing
agent factory and unified systems for seamless integration.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import structlog
from app.agents.factory import AgentBuilderFactory, AgentBuilderConfig, AgentType, AgentTemplate, MemoryType
from app.agents.base.agent import AgentCapability
from app.llm.manager import LLMProviderManager
from app.llm.models import LLMConfig, ProviderType
from app.memory.unified_memory_system import UnifiedMemorySystem
from app.rag.core.unified_rag_system import UnifiedRAGSystem
from app.agents.autonomous.meme_lord_supreme_agent import (
    MemeLordSupremeAgent, 
    MemeEmpireConfig, 
    MemeEmpireStrategy,
    create_meme_lord_supreme_agent
)

logger = structlog.get_logger(__name__)


class MemeEmpireTemplate:
    """Template configurations for different meme empire strategies."""
    
    VIRAL_DOMINATION = AgentBuilderConfig(
        name="Meme Lord Supreme - Viral Domination",
        description="Ultimate meme agent focused on viral domination and maximum reach",
        agent_type=AgentType.AUTONOMOUS,
        memory_type=MemoryType.ADVANCED,
        capabilities=[
            AgentCapability.REASONING,
            AgentCapability.TOOL_USE,
            AgentCapability.MEMORY,
            AgentCapability.PLANNING,
            AgentCapability.LEARNING,
            AgentCapability.MULTIMODAL,
            AgentCapability.VISION
        ],
        tools=[
            "meme_generation",
            "meme_analysis", 
            "meme_collection",
            "web_research",
            "api_integration",
            "database_operations",
            "text_processing_nlp",
            "notification_alert"
        ],
        enable_memory=True,
        enable_learning=True,
        enable_collaboration=True,
        custom_config={
            "empire_strategy": "viral_domination",
            "daily_meme_target": 50,
            "viral_prediction_threshold": 0.8,
            "quality_threshold": 0.85,
            "enable_monetization": True,
            "competitor_monitoring": True
        }
    )
    
    QUALITY_CURATION = AgentBuilderConfig(
        name="Meme Lord Supreme - Quality Curator",
        description="Premium meme agent focused on high-quality content and brand building",
        agent_type=AgentType.AUTONOMOUS,
        memory_type=MemoryType.ADVANCED,
        capabilities=[
            AgentCapability.REASONING,
            AgentCapability.TOOL_USE,
            AgentCapability.MEMORY,
            AgentCapability.PLANNING,
            AgentCapability.LEARNING,
            AgentCapability.MULTIMODAL,
            AgentCapability.VISION
        ],
        tools=[
            "meme_generation",
            "meme_analysis",
            "meme_collection", 
            "web_research",
            "text_processing_nlp"
        ],
        enable_memory=True,
        enable_learning=True,
        custom_config={
            "empire_strategy": "quality_curation",
            "daily_meme_target": 20,
            "viral_prediction_threshold": 0.9,
            "quality_threshold": 0.95,
            "enable_monetization": True,
            "competitor_monitoring": False
        }
    )
    
    MONETIZATION_FOCUS = AgentBuilderConfig(
        name="Meme Lord Supreme - Revenue Generator",
        description="Business-focused meme agent optimized for maximum revenue generation",
        agent_type=AgentType.AUTONOMOUS,
        memory_type=MemoryType.ADVANCED,
        capabilities=[
            AgentCapability.REASONING,
            AgentCapability.TOOL_USE,
            AgentCapability.MEMORY,
            AgentCapability.PLANNING,
            AgentCapability.LEARNING,
            AgentCapability.COLLABORATION
        ],
        tools=[
            "meme_generation",
            "meme_analysis",
            "web_research",
            "api_integration",
            "database_operations",
            "notification_alert"
        ],
        enable_memory=True,
        enable_learning=True,
        enable_collaboration=True,
        custom_config={
            "empire_strategy": "monetization_focus",
            "daily_meme_target": 30,
            "viral_prediction_threshold": 0.75,
            "quality_threshold": 0.8,
            "enable_monetization": True,
            "competitor_monitoring": True
        }
    )


async def register_meme_lord_supreme_with_factory():
    """Register the Meme Lord Supreme Agent with the agent factory."""
    try:
        logger.info("🏭 REGISTERING MEME LORD SUPREME WITH AGENT FACTORY...")
        
        # Initialize systems
        llm_manager = LLMProviderManager()
        await llm_manager.initialize()
        
        unified_memory = UnifiedMemorySystem()
        await unified_memory.initialize()
        
        unified_rag = UnifiedRAGSystem()
        await unified_rag.initialize()
        
        # Create agent factory
        factory = AgentBuilderFactory(llm_manager, unified_memory)
        
        # Add custom builder for Meme Lord Supreme
        async def build_meme_lord_supreme(config: AgentBuilderConfig, llm):
            """Custom builder for Meme Lord Supreme Agent."""
            
            # Extract custom config
            custom_config = config.custom_config or {}
            
            # Create meme empire config
            empire_config = MemeEmpireConfig(
                agent_id=f"meme_lord_{config.name.lower().replace(' ', '_')}",
                empire_name=config.name,
                primary_strategy=MemeEmpireStrategy(custom_config.get("empire_strategy", "viral_domination")),
                daily_meme_target=custom_config.get("daily_meme_target", 50),
                viral_prediction_threshold=custom_config.get("viral_prediction_threshold", 0.8),
                quality_threshold=custom_config.get("quality_threshold", 0.85),
                enable_monetization=custom_config.get("enable_monetization", True),
                competitor_monitoring=custom_config.get("competitor_monitoring", True)
            )
            
            # Create the agent
            agent = MemeLordSupremeAgent(
                llm=llm,
                config=empire_config,
                unified_memory=unified_memory,
                unified_rag=unified_rag
            )
            
            # Initialize the empire
            await agent.initialize_empire()
            
            return agent
        
        # Register the custom builder
        factory._agent_builders[AgentType.AUTONOMOUS] = build_meme_lord_supreme
        
        logger.info("✅ MEME LORD SUPREME REGISTERED WITH FACTORY!")
        
        return factory
        
    except Exception as e:
        logger.error(f"Failed to register Meme Lord Supreme: {str(e)}")
        raise


async def create_meme_empire_showcase():
    """Create a showcase of different meme empire configurations."""
    try:
        print("🎪 CREATING MEME EMPIRE SHOWCASE...")
        print("="*60)
        
        # Initialize systems
        llm_manager = LLMProviderManager()
        await llm_manager.initialize()
        
        llm_config = LLMConfig(
            provider=ProviderType.OLLAMA,
            model_id="llama3.2:latest",  # Required field
            model_name="llama3.2:latest",
            temperature=0.8,
            max_tokens=2000
        )
        
        llm = await llm_manager.create_llm_instance(llm_config)
        
        unified_memory = UnifiedMemorySystem()
        await unified_memory.initialize()
        
        unified_rag = UnifiedRAGSystem()
        await unified_rag.initialize()
        
        # Create different empire configurations
        empires = []
        
        print("\n1. 🚀 VIRAL DOMINATION EMPIRE")
        viral_empire = await create_meme_lord_supreme_agent(
            llm=llm,
            empire_name="Viral Domination Empire",
            strategy=MemeEmpireStrategy.VIRAL_DOMINATION,
            daily_target=50,
            unified_memory=unified_memory,
            unified_rag=unified_rag
        )
        empires.append(("Viral Domination", viral_empire))
        print("   ✅ Created - Focus: Maximum viral reach and engagement")
        
        print("\n2. ⭐ QUALITY CURATION EMPIRE")
        quality_empire = await create_meme_lord_supreme_agent(
            llm=llm,
            empire_name="Premium Quality Empire",
            strategy=MemeEmpireStrategy.QUALITY_CURATION,
            daily_target=20,
            unified_memory=unified_memory,
            unified_rag=unified_rag
        )
        empires.append(("Quality Curation", quality_empire))
        print("   ✅ Created - Focus: Premium content and brand building")
        
        print("\n3. 💰 MONETIZATION EMPIRE")
        money_empire = await create_meme_lord_supreme_agent(
            llm=llm,
            empire_name="Revenue Generation Empire",
            strategy=MemeEmpireStrategy.MONETIZATION_FOCUS,
            daily_target=30,
            unified_memory=unified_memory,
            unified_rag=unified_rag
        )
        empires.append(("Monetization Focus", money_empire))
        print("   ✅ Created - Focus: Maximum revenue generation")
        
        print("\n🎯 EMPIRE SHOWCASE COMPLETE!")
        print(f"   Total Empires Created: {len(empires)}")
        print("   All empires are now running autonomously!")
        
        # Display status of all empires
        print("\n📊 EMPIRE STATUS SUMMARY:")
        print("-" * 60)
        
        for name, empire in empires:
            status = await empire.get_empire_status()
            print(f"\n{name}:")
            print(f"   Strategy: {status['empire_info']['strategy']}")
            print(f"   Empire Rank: {status.get('empire_rank', 'Calculating...')}")
            print(f"   Performance Score: {status['performance_scores'].get('empire_score', 0.0):.2f}")
        
        print("\n🎉 ALL MEME EMPIRES ARE OPERATIONAL!")
        print("   Your meme domination has begun!")
        
        return empires
        
    except Exception as e:
        logger.error(f"Showcase creation failed: {str(e)}")
        raise


async def test_meme_lord_integration():
    """Test the integration of Meme Lord Supreme with existing systems."""
    try:
        print("🧪 TESTING MEME LORD SUPREME INTEGRATION...")
        print("="*50)
        
        # Register with factory
        factory = await register_meme_lord_supreme_with_factory()
        
        # Test building agent through factory
        print("\n1. Testing Factory Integration...")
        config = MemeEmpireTemplate.VIRAL_DOMINATION
        agent = await factory.build_agent(config)
        
        if isinstance(agent, MemeLordSupremeAgent):
            print("   ✅ Factory integration successful!")
        else:
            print("   ❌ Factory integration failed!")
            return False
        
        # Test agent capabilities
        print("\n2. Testing Agent Capabilities...")
        status = await agent.get_empire_status()
        
        if status and 'empire_info' in status:
            print("   ✅ Agent status retrieval successful!")
        else:
            print("   ❌ Agent status retrieval failed!")
            return False
        
        # Test campaign creation
        print("\n3. Testing Campaign Creation...")
        campaign = await agent.create_viral_campaign(
            campaign_name="Integration Test Campaign",
            target_topic="testing",
            duration_hours=1
        )
        
        if campaign and campaign.get('success'):
            print("   ✅ Campaign creation successful!")
        else:
            print("   ❌ Campaign creation failed!")
            return False
        
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("   The Meme Lord Supreme Agent is fully integrated!")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        print(f"   ❌ Integration test error: {str(e)}")
        return False


def main():
    """Main entry point for registration script."""
    print("🎭👑 MEME LORD SUPREME REGISTRATION SYSTEM 👑🎭")
    print()
    print("Choose an option:")
    print("1. Register with Agent Factory")
    print("2. Create Empire Showcase")
    print("3. Test Integration")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🏭 REGISTERING WITH AGENT FACTORY...")
        asyncio.run(register_meme_lord_supreme_with_factory())
        print("✅ Registration complete!")
        
    elif choice == "2":
        print("\n🎪 CREATING EMPIRE SHOWCASE...")
        asyncio.run(create_meme_empire_showcase())
        
    elif choice == "3":
        print("\n🧪 TESTING INTEGRATION...")
        success = asyncio.run(test_meme_lord_integration())
        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            
    elif choice == "4":
        print("\n👋 Goodbye! Your meme empire awaits!")
        
    else:
        print("\n❌ Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()
