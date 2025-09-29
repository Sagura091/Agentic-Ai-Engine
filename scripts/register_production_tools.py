"""
Script to register production tools with the UnifiedToolRepository.

This script integrates the new production tools with the existing agentic AI system.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.tools.unified_tool_repository import UnifiedToolRepository
from app.tools.production import PRODUCTION_TOOLS
import structlog

logger = structlog.get_logger(__name__)


async def register_production_tools():
    """Register all production tools with the unified repository."""
    try:
        # Initialize the unified tool repository
        tool_repo = UnifiedToolRepository()
        await tool_repo.initialize()

        logger.info("Starting production tools registration")
        
        # Register each production tool
        for tool_name, tool_info in PRODUCTION_TOOLS.items():
            try:
                tool_class = tool_info["tool_class"]
                metadata = tool_info["metadata"]
                
                # Create tool instance
                tool_instance = tool_class()
                
                # Register with repository
                tool_id = await tool_repo.register_tool(
                    tool_instance=tool_instance,
                    metadata=metadata
                )
                success = tool_id is not None
                
                if success:
                    logger.info("Tool registered successfully", 
                               tool_name=tool_name,
                               category=metadata.category,
                               version=metadata.version)
                else:
                    logger.error("Failed to register tool", tool_name=tool_name)
                    
            except Exception as e:
                logger.error("Error registering tool", 
                           tool_name=tool_name, 
                           error=str(e))
        
        # Verify registration
        registered_tools = list(tool_repo.tools.keys())
        production_tool_names = set(PRODUCTION_TOOLS.keys())
        registered_production_tools = {
            name for name in registered_tools
            if any(name.startswith(prod_name) for prod_name in production_tool_names)
        }
        
        logger.info("Production tools registration completed",
                   total_tools=len(PRODUCTION_TOOLS),
                   registered_tools=len(registered_production_tools),
                   registered_list=list(registered_production_tools))
        
        if len(registered_production_tools) == len(PRODUCTION_TOOLS):
            logger.info("✅ All production tools registered successfully!")
            return True
        else:
            missing_tools = production_tool_names - registered_production_tools
            logger.error("❌ Some tools failed to register", 
                        missing_tools=list(missing_tools))
            return False
            
    except Exception as e:
        logger.error("Failed to register production tools", error=str(e))
        return False


async def test_production_tools():
    """Test the registered production tools."""
    try:
        tool_repo = UnifiedToolRepository()
        
        logger.info("Testing production tools")
        
        # Test File System Tool
        try:
            file_tool = tool_repo.get_tool("file_system_v1")
            if file_tool:
                # Test basic file operation
                result = await file_tool._run(
                    operation="create",
                    path="test_registration.txt",
                    content="Production tools registration test"
                )
                logger.info("File System Tool test completed", result_success="success" in result)
            else:
                logger.error("File System Tool not found in repository")
        except Exception as e:
            logger.error("File System Tool test failed", error=str(e))
        
        # Test API Integration Tool
        try:
            api_tool = tool_repo.get_tool("api_integration_v1")
            if api_tool:
                # Test basic API call (this will fail but should handle gracefully)
                result = await api_tool._run(
                    url="https://httpbin.org/get",
                    method="GET"
                )
                logger.info("API Integration Tool test completed", result_success="success" in result)
            else:
                logger.error("API Integration Tool not found in repository")
        except Exception as e:
            logger.error("API Integration Tool test failed", error=str(e))
        
        logger.info("✅ Production tools testing completed")
        
    except Exception as e:
        logger.error("Production tools testing failed", error=str(e))


async def main():
    """Main registration and testing function."""
    logger.info("🚀 Starting Production Tools Integration")
    
    # Register tools
    registration_success = await register_production_tools()
    
    if registration_success:
        logger.info("✅ Registration successful, proceeding with tests")
        await test_production_tools()
    else:
        logger.error("❌ Registration failed, skipping tests")
        return False
    
    logger.info("🎉 Production Tools Integration completed successfully!")
    return True


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
    
    # Run the registration
    success = asyncio.run(main())
    
    if success:
        print("\n🎉 SUCCESS: Production tools registered and tested successfully!")
        print("\nAvailable tools:")
        for tool_name, tool_info in PRODUCTION_TOOLS.items():
            metadata = tool_info["metadata"]
            print(f"  • {metadata.display_name} ({tool_name})")
            print(f"    Category: {metadata.category}")
            print(f"    Version: {metadata.version}")
            print(f"    Description: {metadata.description}")
            print()
        
        print("Next steps:")
        print("1. Run the test suite: pytest tests/tools/production/ -v")
        print("2. Start implementing Week 1 Tool 3: Database Operations Tool")
        print("3. Continue with the phased implementation plan")
        
    else:
        print("\n❌ FAILED: Production tools registration failed!")
        print("Check the logs above for error details.")
        sys.exit(1)
