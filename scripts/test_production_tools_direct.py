"""
Direct test of production tools without repository integration.

This script tests the tools directly to ensure they work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.tools.production.file_system_tool import FileSystemTool
from app.tools.production.api_integration_tool import APIIntegrationTool
import structlog

logger = structlog.get_logger(__name__)


async def test_file_system_tool():
    """Test the File System Tool directly."""
    try:
        logger.info("Testing File System Tool")
        
        # Create tool instance
        tool = FileSystemTool()
        
        # Test file creation
        result = await tool._run(
            operation="create",
            path="test_direct.txt",
            content="Direct test of file system tool"
        )
        
        logger.info("File creation test", result=result[:100] + "..." if len(result) > 100 else result)
        
        # Test file reading
        result = await tool._run(
            operation="read",
            path="test_direct.txt"
        )
        
        logger.info("File reading test", result=result[:100] + "..." if len(result) > 100 else result)
        
        # Test directory listing
        result = await tool._run(
            operation="list",
            path="."
        )
        
        logger.info("Directory listing test", result=result[:100] + "..." if len(result) > 100 else result)
        
        logger.info("✅ File System Tool tests completed successfully")
        return True
        
    except Exception as e:
        logger.error("❌ File System Tool test failed", error=str(e))
        return False


async def test_api_integration_tool():
    """Test the API Integration Tool directly."""
    try:
        logger.info("Testing API Integration Tool")
        
        # Create tool instance
        tool = APIIntegrationTool()
        
        # Test simple GET request to a public API
        result = await tool._run(
            url="https://httpbin.org/get",
            method="GET"
        )
        
        logger.info("GET request test", result=result[:200] + "..." if len(result) > 200 else result)
        
        # Test POST request
        result = await tool._run(
            url="https://httpbin.org/post",
            method="POST",
            json_data={"test": "data", "timestamp": "2024-01-01"}
        )
        
        logger.info("POST request test", result=result[:200] + "..." if len(result) > 200 else result)
        
        logger.info("✅ API Integration Tool tests completed successfully")
        return True
        
    except Exception as e:
        logger.error("❌ API Integration Tool test failed", error=str(e))
        return False


async def main():
    """Main testing function."""
    logger.info("🚀 Starting Direct Production Tools Testing")
    
    # Test File System Tool
    fs_success = await test_file_system_tool()
    
    # Test API Integration Tool
    api_success = await test_api_integration_tool()
    
    if fs_success and api_success:
        logger.info("🎉 All production tools tests passed!")
        return True
    else:
        logger.error("❌ Some production tools tests failed")
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
    
    if success:
        print("\n🎉 SUCCESS: Production tools work correctly!")
        print("\nTool Status:")
        print("  ✅ File System Operations Tool - WORKING")
        print("  ✅ API Integration Tool - WORKING")
        print("\nNext Steps:")
        print("1. Tools are ready for integration with agents")
        print("2. Continue with Week 1 remaining tools (Database, Text Processing)")
        print("3. Run full integration tests when ready")
        
    else:
        print("\n❌ FAILED: Some production tools have issues!")
        print("Check the logs above for error details.")
        sys.exit(1)
