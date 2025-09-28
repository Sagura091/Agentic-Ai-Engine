#!/usr/bin/env python3
"""
Tool System Diagnostic Script
Analyzes the current state of the tool system and identifies issues.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add the app directory to Python path
sys.path.append('.')

async def diagnose_tools():
    """Comprehensive tool system diagnostic."""
    try:
        print('🔍 TOOL SYSTEM DIAGNOSTIC')
        print('=' * 50)
        
        # Import system components
        from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
        
        # Get orchestrator
        orchestrator = get_enhanced_system_orchestrator()
        if not orchestrator:
            print('❌ ERROR: Could not get system orchestrator')
            return
            
        print(f'✅ System orchestrator available: {type(orchestrator).__name__}')
        
        # Check if initialized
        if not orchestrator.is_initialized:
            print('⚠️  System not initialized, initializing now...')
            await orchestrator.initialize()
        
        print(f'✅ System initialized: {orchestrator.is_initialized}')
        
        # Check tool repository
        if not orchestrator.tool_repository:
            print('❌ ERROR: Tool repository not available')
            return
            
        print('✅ Tool repository available')
        
        # Get tool stats
        stats = orchestrator.tool_repository.stats
        print(f'📊 Total tools registered: {stats["total_tools"]}')
        print(f'📊 RAG-enabled tools: {stats["rag_enabled_tools"]}')
        
        # Check for revolutionary web scraper
        tools = orchestrator.tool_repository.tools
        if 'revolutionary_web_scraper' in tools:
            print('✅ Revolutionary Web Scraper tool is registered')
            scraper_tool = tools['revolutionary_web_scraper']
            print(f'   Tool name: {scraper_tool.name}')
            print(f'   Tool description: {scraper_tool.description[:100]}...')
        else:
            print('❌ Revolutionary Web Scraper tool NOT found')
            
        # List all available tools
        print(f'\n📋 Available tools ({len(tools)}):')
        for tool_id, tool in tools.items():
            print(f'   - {tool_id}: {tool.name}')
            
        # Test web scraper capabilities
        if 'revolutionary_web_scraper' in tools:
            print('\n🌐 TESTING WEB SCRAPER CAPABILITIES')
            print('-' * 40)
            scraper = tools['revolutionary_web_scraper']
            
            # Test basic functionality
            try:
                # Simple search test
                result = await scraper._arun(
                    action="search",
                    query="python programming",
                    search_engines=["duckduckgo"],
                    num_results=3,
                    scraping_mode="fast"
                )
                print('✅ Web scraper search test: PASSED')
                print(f'   Result length: {len(result)} characters')
            except Exception as e:
                print(f'❌ Web scraper search test: FAILED - {str(e)}')
                
        # Check agent tool assignment
        print('\n👤 TESTING AGENT TOOL ASSIGNMENT')
        print('-' * 40)
        
        try:
            from app.agents.factory import AgentBuilderFactory, AgentBuilderConfig, AgentType
            from app.agents.base.agent import AgentCapability
            from app.llm.models import LLMConfig, ProviderType
            from app.llm.manager import get_enhanced_llm_manager

            # Get LLM manager
            llm_manager = get_enhanced_llm_manager()

            # Create a test agent configuration
            config = AgentBuilderConfig(
                name="test_agent",
                description="Test agent for tool assignment",
                agent_type=AgentType.REACT,
                llm_config=LLMConfig(provider=ProviderType.OLLAMA, model_id="llama3.2:latest"),
                capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE],
                tools=["revolutionary_web_scraper", "calculator"]
            )

            factory = AgentBuilderFactory(llm_manager=llm_manager)
            
            # Test tool retrieval
            test_tools = await factory._get_agent_tools(config.tools)
            print(f'✅ Agent tool assignment test: Retrieved {len(test_tools)} tools')
            for tool in test_tools:
                print(f'   - {tool.name}')
                
        except Exception as e:
            print(f'❌ Agent tool assignment test: FAILED - {str(e)}')
            
    except Exception as e:
        print(f'❌ DIAGNOSTIC ERROR: {str(e)}')
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_tools())
