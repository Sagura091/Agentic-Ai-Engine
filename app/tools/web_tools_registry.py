"""
Web Tools Registry for Agentic AI System.

This module registers and manages web-related tools for agents,
including the advanced web research tool and other web utilities.
"""

import asyncio
import structlog
from typing import Dict, List, Any

from app.tools.web_research_tool import WebResearchTool, WEB_RESEARCH_TOOL_METADATA
from app.tools.production_tool_system import production_tool_registry
from app.tools.dynamic_tool_factory import ToolCategory

logger = structlog.get_logger(__name__)


class WebToolsRegistry:
    """Registry for web-related tools."""
    
    def __init__(self):
        self.registered_tools: Dict[str, Any] = {}
        self.tool_instances: Dict[str, Any] = {}
        
    async def initialize_web_tools(self) -> None:
        """Initialize and register all web tools."""
        try:
            logger.info("Initializing web tools registry")
            
            # Register the advanced web research tool
            await self._register_web_research_tool()
            
            # Register additional web tools
            await self._register_additional_web_tools()
            
            logger.info(
                "Web tools registry initialized successfully",
                total_tools=len(self.registered_tools)
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize web tools: {str(e)}")
            raise
    
    async def _register_web_research_tool(self) -> None:
        """Register the advanced web research tool."""
        try:
            # Create tool instance
            web_research_tool = WebResearchTool()
            
            # Register with production tool registry
            await production_tool_registry.register_production_tool(
                web_research_tool,
                metadata=WEB_RESEARCH_TOOL_METADATA
            )
            
            # Store in local registry
            self.registered_tools["web_research"] = web_research_tool
            self.tool_instances["web_research"] = web_research_tool
            
            logger.info("Web research tool registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register web research tool: {str(e)}")
            raise
    
    async def _register_additional_web_tools(self) -> None:
        """Register additional web utility tools."""
        try:
            # URL validator tool
            url_validator = await self._create_url_validator_tool()
            await production_tool_registry.register_production_tool(
                url_validator,
                metadata={
                    "category": ToolCategory.WEB_SCRAPING.value,
                    "description": "Validate and analyze URLs",
                    "safety_level": "safe"
                }
            )
            self.registered_tools["url_validator"] = url_validator
            
            # Website monitor tool
            website_monitor = await self._create_website_monitor_tool()
            await production_tool_registry.register_production_tool(
                website_monitor,
                metadata={
                    "category": ToolCategory.WEB_SCRAPING.value,
                    "description": "Monitor websites for changes",
                    "safety_level": "safe"
                }
            )
            self.registered_tools["website_monitor"] = website_monitor
            
            logger.info("Additional web tools registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register additional web tools: {str(e)}")
    
    async def _create_url_validator_tool(self):
        """Create URL validator tool."""
        from langchain_core.tools import tool
        from urllib.parse import urlparse
        from app.http_client import SimpleHTTPClient

        @tool
        async def validate_url(url: str) -> str:
            """
            Validate a URL and check if it's accessible.

            Args:
                url: URL to validate

            Returns:
                Validation result with status and metadata
            """
            try:
                # Parse URL
                parsed = urlparse(url)

                if not parsed.scheme or not parsed.netloc:
                    return f"Invalid URL format: {url}"

                # Check accessibility
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                async with SimpleHTTPClient(base_url, timeout=10) as client:
                    try:
                        response = await client.head("/")
                        content_type = response.headers.get('content-type', 'unknown')
                        return f"URL is valid and accessible. Status: {response.status_code}, Content-Type: {content_type}"
                    except Exception as e:
                        return f"URL is valid but not accessible: {str(e)}"
                        
            except Exception as e:
                return f"URL validation failed: {str(e)}"
        
        return validate_url
    
    async def _create_website_monitor_tool(self):
        """Create website monitoring tool."""
        from langchain_core.tools import tool
        import hashlib
        from urllib.parse import urlparse
        from app.http_client import SimpleHTTPClient
        from datetime import datetime

        # Store for monitoring data
        monitoring_data = {}

        @tool
        async def monitor_website(url: str, check_interval: int = 3600) -> str:
            """
            Monitor a website for changes.

            Args:
                url: URL to monitor
                check_interval: Check interval in seconds

            Returns:
                Monitoring status and change detection results
            """
            try:
                parsed = urlparse(url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                path = parsed.path or "/"
                if parsed.query:
                    path += f"?{parsed.query}"

                async with SimpleHTTPClient(base_url, timeout=30) as client:
                    response = await client.get(path)
                    content = response.text
                    content_hash = hashlib.md5(content.encode()).hexdigest()

                    current_time = datetime.utcnow()

                    if url in monitoring_data:
                        previous_hash = monitoring_data[url]['hash']
                        previous_time = monitoring_data[url]['last_check']

                        if content_hash != previous_hash:
                            monitoring_data[url] = {
                                'hash': content_hash,
                                'last_check': current_time,
                                'last_change': current_time
                            }
                            return f"Website changed! URL: {url}, Last change: {current_time.isoformat()}"
                        else:
                            monitoring_data[url]['last_check'] = current_time
                            return f"No changes detected. URL: {url}, Last checked: {current_time.isoformat()}"
                    else:
                        monitoring_data[url] = {
                            'hash': content_hash,
                            'last_check': current_time,
                            'last_change': current_time
                        }
                        return f"Started monitoring: {url}, Initial check: {current_time.isoformat()}"
                            
            except Exception as e:
                return f"Website monitoring failed for {url}: {str(e)}"
        
        return monitor_website
    
    def get_tool(self, tool_name: str):
        """Get a registered web tool by name."""
        return self.registered_tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered web tools."""
        return list(self.registered_tools.keys())
    
    async def cleanup(self):
        """Clean up tool resources."""
        for tool_name, tool_instance in self.tool_instances.items():
            if hasattr(tool_instance, 'cleanup'):
                try:
                    await tool_instance.cleanup()
                    logger.info(f"Cleaned up tool: {tool_name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup tool {tool_name}: {str(e)}")


# Global web tools registry instance
web_tools_registry = WebToolsRegistry()


async def initialize_web_tools():
    """Initialize web tools for the system."""
    await web_tools_registry.initialize_web_tools()


async def get_web_research_tool():
    """Get the web research tool instance."""
    return web_tools_registry.get_tool("web_research")


def get_available_web_tools() -> List[str]:
    """Get list of available web tools."""
    return web_tools_registry.list_tools()


# Tool usage examples and documentation
WEB_TOOLS_USAGE_EXAMPLES = {
    "web_research": {
        "search_example": {
            "description": "Search for AI agent research papers",
            "usage": """
            # Search the web for information
            result = await web_research_tool._arun(
                action="search",
                query="AI agent research papers 2024",
                num_results=10,
                search_engine="duckduckgo"
            )
            """
        },
        "scrape_example": {
            "description": "Scrape content from a research paper website",
            "usage": """
            # Scrape webpage content
            result = await web_research_tool._arun(
                action="scrape",
                url="https://arxiv.org/abs/2401.12345",
                extract_text=True,
                extract_links=True,
                css_selector="div.abstract"
            )
            """
        },
        "multi_scrape_example": {
            "description": "Scrape multiple URLs for comprehensive research",
            "usage": """
            # Scrape multiple URLs
            result = await web_research_tool._arun(
                action="multi_scrape",
                urls=[
                    "https://example1.com/article",
                    "https://example2.com/research",
                    "https://example3.com/data"
                ],
                extract_text=True,
                extract_links=False
            )
            """
        }
    },
    "url_validator": {
        "example": {
            "description": "Validate URL accessibility",
            "usage": """
            # Validate a URL
            result = await url_validator.arun("https://example.com")
            """
        }
    },
    "website_monitor": {
        "example": {
            "description": "Monitor website for changes",
            "usage": """
            # Monitor a website
            result = await website_monitor.arun("https://news.example.com")
            """
        }
    }
}


# Integration with agent system
async def add_web_tools_to_agent(agent):
    """Add all web tools to an agent."""
    try:
        for tool_name, tool in web_tools_registry.registered_tools.items():
            await agent.add_tool(tool)
            logger.info(f"Added web tool {tool_name} to agent {agent.agent_id}")
    except Exception as e:
        logger.error(f"Failed to add web tools to agent: {str(e)}")


# Export key components
__all__ = [
    "web_tools_registry",
    "initialize_web_tools", 
    "get_web_research_tool",
    "get_available_web_tools",
    "add_web_tools_to_agent",
    "WEB_TOOLS_USAGE_EXAMPLES"
]
