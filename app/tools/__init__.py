"""
Unified Tool System for Multi-Agent Architecture.

This module provides the unified tool repository and management system
for the multi-agent architecture, enabling centralized tool access
with agent-specific capabilities and access controls.

Key Features:
- Unified tool repository with centralized management
- Agent-specific tool profiles and capabilities
- Access control and permission management
- Performance tracking and usage analytics
- Tool categorization and metadata management
- Seamless integration with agent isolation system

Components:
- UnifiedToolRepository: Central tool repository and manager
- AgentToolProfile: Agent-specific tool access and capabilities
- ToolMetadata: Comprehensive metadata tracking for tools
- Built-in tools for common agent operations
"""

# Import unified tool repository (primary system)
from .unified_tool_repository import (
    UnifiedToolRepository,
    ToolCategory,
    ToolAccessLevel,
    ToolCapability,
    ToolMetadata,
    AgentToolProfile
)

# Import built-in tools
from .calculator_tool import calculator_tool
from .business_intelligence_tool import business_intelligence_tool

__all__ = [
    # Unified Tool System
    "UnifiedToolRepository",
    "ToolCategory",
    "ToolAccessLevel",
    "ToolCapability",
    "ToolMetadata",
    "AgentToolProfile",

    # Built-in Tools
    "calculator_tool",
    "business_intelligence_tool"
]

# Version information
__version__ = "1.0.0"
__author__ = "Agentic AI Team"
__description__ = "Unified tool system for multi-agent capabilities"

# Module metadata
TOOL_SYSTEM_FEATURES = [
    "unified_tool_repository",
    "agent_specific_profiles",
    "access_control_management",
    "performance_tracking",
    "usage_analytics",
    "seamless_agent_integration",
    "comprehensive_categorization",
    "metadata_management"
]

SUPPORTED_TOOL_CATEGORIES = [
    "general",
    "web_scraping", 
    "file_operations",
    "data_processing",
    "api_integration",
    "computation",
    "communication",
    "automation",
    "analysis",
    "creative",
    "research",
    "monitoring",
    "security",
    "database",
    "custom"
]

BUILT_IN_TEMPLATES = [
    "web_scraper",
    "file_reader",
    "api_caller",
    "json_processor"
]

# Configuration defaults
DEFAULT_TOOL_CONFIG = {
    "category": "custom",
    "complexity": "simple",
    "safety_level": "safe",
    "enable_metrics": True,
    "enable_caching": False,
    "timeout_seconds": 30,
    "max_retries": 3
}

def get_available_templates():
    """Get list of available tool templates."""
    return BUILT_IN_TEMPLATES.copy()

def get_tool_categories():
    """Get list of supported tool categories."""
    return SUPPORTED_TOOL_CATEGORIES.copy()

def get_system_info():
    """Get comprehensive information about the tool system."""
    return {
        "version": __version__,
        "description": __description__,
        "features": TOOL_SYSTEM_FEATURES,
        "categories": SUPPORTED_TOOL_CATEGORIES,
        "templates": BUILT_IN_TEMPLATES,
        "unified_repository_enabled": True
    }

# Export module information
__module_info__ = {
    "name": "tools",
    "version": __version__,
    "description": __description__,
    "features": TOOL_SYSTEM_FEATURES
}
