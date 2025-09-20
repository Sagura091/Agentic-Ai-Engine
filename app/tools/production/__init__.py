"""
Production Tools Module - Week 1 Implementation.

This module contains the first batch of revolutionary AI agent tools:
- File System Operations Tool
- API Integration Tool

These tools provide the foundation for all other tools and agent operations.
"""

from .file_system_tool import FileSystemTool, file_system_tool, FILE_SYSTEM_TOOL_METADATA
from .api_integration_tool import APIIntegrationTool, api_integration_tool, API_INTEGRATION_TOOL_METADATA

__all__ = [
    "FileSystemTool",
    "file_system_tool",
    "FILE_SYSTEM_TOOL_METADATA",
    "APIIntegrationTool",
    "api_integration_tool",
    "API_INTEGRATION_TOOL_METADATA"
]

# Tool registry for easy access
PRODUCTION_TOOLS = {
    "file_system": {
        "tool_class": FileSystemTool,
        "metadata": FILE_SYSTEM_TOOL_METADATA
    },
    "api_integration": {
        "tool_class": APIIntegrationTool,
        "metadata": API_INTEGRATION_TOOL_METADATA
    }
}

def get_production_tool(tool_name: str):
    """Get production tool by name."""
    if tool_name not in PRODUCTION_TOOLS:
        raise ValueError(f"Unknown production tool: {tool_name}")
    
    tool_info = PRODUCTION_TOOLS[tool_name]
    return tool_info["tool_class"](), tool_info["metadata"]

def list_production_tools():
    """List all available production tools."""
    return list(PRODUCTION_TOOLS.keys())
