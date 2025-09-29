"""
Test Production Tools Integration with Backend System.

This script tests that the production tools are properly registered
and working with the unified tool repository and agent system.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.unified_system_orchestrator import UnifiedSystemOrchestrator
from app.tools.unified_tool_repository import UnifiedToolRepository
import structlog

logger = structlog.get_logger(__name__)


async def test_tool_registration():
    """Test that production tools are properly registered."""
    try:
        logger.info("🔧 Testing tool registration")
        
        # Create orchestrator (this will register all tools)
        orchestrator = UnifiedSystemOrchestrator()
        await orchestrator.initialize()
        
        # Get tool repository
        tool_repo = orchestrator.tool_repository
        
        # Check if our production tools are registered
        tools = tool_repo.tools
        tool_names = list(tools.keys())
        
        logger.info("📋 Registered tools", tools=tool_names)
        
        # Check for our specific tools
        expected_tools = ["file_system", "api_integration"]
        found_tools = []
        
        for tool_name in expected_tools:
            if tool_name in tools:
                found_tools.append(tool_name)
                logger.info("✅ Found production tool", tool=tool_name)
            else:
                logger.error("❌ Missing production tool", tool=tool_name)
        
        if len(found_tools) == len(expected_tools):
            logger.info("🎉 All production tools registered successfully!")
            return True
        else:
            logger.error("❌ Some production tools missing")
            return False
            
    except Exception as e:
        logger.error("❌ Tool registration test failed", error=str(e))
        return False


async def test_file_system_tool():
    """Test file system tool through the repository."""
    try:
        logger.info("📁 Testing File System Tool integration")
        
        # Create orchestrator
        orchestrator = UnifiedSystemOrchestrator()
        await orchestrator.initialize()
        
        # Get tool repository
        tool_repo = orchestrator.tool_repository
        
        # Get file system tool
        if "file_system" not in tool_repo.tools:
            logger.error("❌ File system tool not found")
            return False
        
        file_tool = tool_repo.tools["file_system"]
        
        # Test file creation
        result = await file_tool._run(
            operation="create",
            path="integration_test.txt",
            content="Integration test successful!"
        )
        
        logger.info("📝 File creation result", result=result[:100] + "..." if len(result) > 100 else result)
        
        # Test file reading
        result = await file_tool._run(
            operation="read",
            path="integration_test.txt"
        )
        
        logger.info("📖 File reading result", result=result[:100] + "..." if len(result) > 100 else result)
        
        logger.info("✅ File System Tool integration test passed!")
        return True
        
    except Exception as e:
        logger.error("❌ File System Tool integration test failed", error=str(e))
        return False


async def test_api_integration_tool():
    """Test API integration tool through the repository."""
    try:
        logger.info("🌐 Testing API Integration Tool integration")
        
        # Create orchestrator
        orchestrator = UnifiedSystemOrchestrator()
        await orchestrator.initialize()
        
        # Get tool repository
        tool_repo = orchestrator.tool_repository
        
        # Get API integration tool
        if "api_integration" not in tool_repo.tools:
            logger.error("❌ API integration tool not found")
            return False
        
        api_tool = tool_repo.tools["api_integration"]
        
        # Test simple GET request
        result = await api_tool._run(
            url="https://httpbin.org/get",
            method="GET"
        )
        
        logger.info("🌐 API request result", result=result[:200] + "..." if len(result) > 200 else result)
        
        logger.info("✅ API Integration Tool integration test passed!")
        return True
        
    except Exception as e:
        logger.error("❌ API Integration Tool integration test failed", error=str(e))
        return False


async def test_tool_repository_stats():
    """Test tool repository statistics."""
    try:
        logger.info("📊 Testing tool repository statistics")
        
        # Create orchestrator
        orchestrator = UnifiedSystemOrchestrator()
        await orchestrator.initialize()
        
        # Get tool repository
        tool_repo = orchestrator.tool_repository
        
        # Get stats
        stats = tool_repo.stats
        
        logger.info("📈 Tool Repository Stats", 
                   total_tools=stats["total_tools"],
                   tools_by_category=stats["tools_by_category"])
        
        # Check that we have our production tools
        if stats["total_tools"] >= 2:  # At least our 2 production tools
            logger.info("✅ Tool repository statistics look good!")
            return True
        else:
            logger.error("❌ Tool repository statistics show missing tools")
            return False
            
    except Exception as e:
        logger.error("❌ Tool repository statistics test failed", error=str(e))
        return False


async def main():
    """Main testing function."""
    logger.info("🚀 Starting Production Tools Integration Testing")
    
    # Test tool registration
    registration_success = await test_tool_registration()
    
    # Test file system tool
    file_system_success = await test_file_system_tool()
    
    # Test API integration tool
    api_integration_success = await test_api_integration_tool()
    
    # Test repository stats
    stats_success = await test_tool_repository_stats()
    
    # Overall result
    all_tests_passed = all([
        registration_success,
        file_system_success,
        api_integration_success,
        stats_success
    ])
    
    if all_tests_passed:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        print("\n🎉 SUCCESS: Production tools are fully integrated!")
        print("\nIntegration Status:")
        print("  ✅ Tool Registration - WORKING")
        print("  ✅ File System Tool - WORKING")
        print("  ✅ API Integration Tool - WORKING")
        print("  ✅ Repository Statistics - WORKING")
        print("\nNext Steps:")
        print("1. Production tools are ready for agent use")
        print("2. Continue with Week 1 remaining tools")
        print("3. Begin Phase 2 implementation")
        return True
    else:
        logger.error("❌ SOME INTEGRATION TESTS FAILED!")
        print("\n❌ FAILED: Some integration tests failed!")
        print("Check the logs above for error details.")
        return False


if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Run the tests
    success = asyncio.run(main())
    
    if not success:
        sys.exit(1)
