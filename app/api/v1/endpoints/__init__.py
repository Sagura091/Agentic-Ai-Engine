"""API endpoints for the Agentic AI Microservice."""

# Import all endpoint modules to make them available for the router
from . import agents, workflows, admin, health, openwebui, standalone, autonomous_agents, enhanced_orchestration, monitoring, models, settings, conversational_agents

__all__ = ["agents", "workflows", "admin", "health", "openwebui", "standalone", "autonomous_agents", "enhanced_orchestration", "monitoring", "models", "settings", "conversational_agents"]
