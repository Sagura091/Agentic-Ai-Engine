"""
Production Tools Module - Week 1 Implementation.

This module contains the first batch of revolutionary AI agent tools:
- File System Operations Tool
- API Integration Tool
- Database Operations Tool
- Text Processing & NLP Tool
- Password & Security Tool
- Notification & Alert Tool
- QR Code & Barcode Tool
- Weather & Environmental Tool

These tools provide comprehensive capabilities for all agent operations.
"""

from .file_system_tool import file_system_tool
from .api_integration_tool import api_integration_tool
from .database_operations_tool import database_operations_tool
from .text_processing_nlp_tool import text_processing_nlp_tool
from .password_security_tool import password_security_tool
from .notification_alert_tool import notification_alert_tool
from .qr_barcode_tool import qr_barcode_tool
from .weather_environmental_tool import weather_environmental_tool
from .screenshot_analysis_tool import screenshot_analysis_tool
from .browser_automation_tool import browser_automation_tool
from .computer_use_agent_tool import computer_use_agent_tool
from .revolutionary_document_intelligence_tool import RevolutionaryDocumentIntelligenceTool

# Create instance of the revolutionary document intelligence tool
revolutionary_document_intelligence_tool = RevolutionaryDocumentIntelligenceTool()

__all__ = [
    "file_system_tool",
    "api_integration_tool",
    "database_operations_tool",
    "text_processing_nlp_tool",
    "password_security_tool",
    "notification_alert_tool",
    "qr_barcode_tool",
    "weather_environmental_tool",
    "screenshot_analysis_tool",
    "browser_automation_tool",
    "computer_use_agent_tool",
    "revolutionary_document_intelligence_tool"
]

# Tool registry for easy access
PRODUCTION_TOOLS = {
    "file_system": file_system_tool,
    "api_integration": api_integration_tool,
    "database_operations": database_operations_tool,
    "text_processing_nlp": text_processing_nlp_tool,
    "password_security": password_security_tool,
    "notification_alert": notification_alert_tool,
    "qr_barcode": qr_barcode_tool,
    "weather_environmental": weather_environmental_tool,
    "screenshot_analysis": screenshot_analysis_tool,
    "browser_automation": browser_automation_tool,
    "computer_use_agent": computer_use_agent_tool,
    "revolutionary_document_intelligence": revolutionary_document_intelligence_tool
}

def get_production_tool(tool_name: str):
    """Get production tool by name."""
    if tool_name not in PRODUCTION_TOOLS:
        raise ValueError(f"Unknown production tool: {tool_name}")

    return PRODUCTION_TOOLS[tool_name]

def list_production_tools():
    """List all available production tools."""
    return list(PRODUCTION_TOOLS.keys())

def get_all_production_tools():
    """Get all production tools as a list."""
    return list(PRODUCTION_TOOLS.values())
