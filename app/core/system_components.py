"""
Modular system components for the Agentic AI platform.

This module breaks down the monolithic orchestrator into focused, single-responsibility
components that can be independently managed and tested.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import asyncio
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ComponentStatus:
    """Status of a system component."""
    name: str
    is_initialized: bool = False
    is_running: bool = False
    is_healthy: bool = False
    last_health_check: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None


class SystemComponent(ABC):
    """Base class for all system components."""
    
    def __init__(self, name: str):
        self.name = name
        self.status = ComponentStatus(name=name)
        self.dependencies: List[str] = []
        self.dependents: List[str] = []
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the component."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the component."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check component health."""
        pass
    
    @abstractmethod
    async def dispose(self):
        """Dispose the component."""
        pass
    
    def add_dependency(self, component_name: str):
        """Add a dependency."""
        if component_name not in self.dependencies:
            self.dependencies.append(component_name)
    
    def add_dependent(self, component_name: str):
        """Add a dependent component."""
        if component_name not in self.dependents:
            self.dependents.append(component_name)


class RAGComponent(SystemComponent):
    """RAG system component."""
    
    def __init__(self):
        super().__init__("RAG")
        self.rag_system = None
    
    async def initialize(self) -> bool:
        """Initialize RAG system."""
        try:
            from app.rag.core.unified_rag_system import UnifiedRAGSystem, UnifiedRAGConfig
            
            config = UnifiedRAGConfig()
            self.rag_system = UnifiedRAGSystem(config)
            await self.rag_system.initialize()
            
            self.status.is_initialized = True
            logger.info("RAG component initialized")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to initialize RAG component: {e}")
            return False
    
    async def start(self) -> bool:
        """Start RAG system."""
        try:
            if not self.status.is_initialized:
                return False
            
            # RAG system is ready to use
            self.status.is_running = True
            logger.info("RAG component started")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to start RAG component: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop RAG system."""
        try:
            self.status.is_running = False
            logger.info("RAG component stopped")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to stop RAG component: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check RAG system health."""
        try:
            if not self.rag_system:
                return False
            
            # Check if RAG system is responsive
            health = await self.rag_system.get_health_status()
            self.status.is_healthy = health.get("status") == "healthy"
            self.status.last_health_check = datetime.utcnow()
            
            return self.status.is_healthy
        except Exception as e:
            self.status.error_message = str(e)
            self.status.is_healthy = False
            logger.error(f"RAG health check failed: {e}")
            return False
    
    async def dispose(self):
        """Dispose RAG system."""
        try:
            if self.rag_system:
                await self.rag_system.dispose()
            self.status.is_initialized = False
            self.status.is_running = False
            logger.info("RAG component disposed")
        except Exception as e:
            logger.error(f"Failed to dispose RAG component: {e}")


class MemoryComponent(SystemComponent):
    """Memory system component."""
    
    def __init__(self):
        super().__init__("Memory")
        self.memory_system = None
    
    async def initialize(self) -> bool:
        """Initialize memory system."""
        try:
            from app.memory.unified_memory_system import UnifiedMemorySystem
            
            self.memory_system = UnifiedMemorySystem()
            await self.memory_system.initialize()
            
            self.status.is_initialized = True
            logger.info("Memory component initialized")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to initialize Memory component: {e}")
            return False
    
    async def start(self) -> bool:
        """Start memory system."""
        try:
            if not self.status.is_initialized:
                return False
            
            self.status.is_running = True
            logger.info("Memory component started")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to start Memory component: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop memory system."""
        try:
            self.status.is_running = False
            logger.info("Memory component stopped")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to stop Memory component: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check memory system health."""
        try:
            if not self.memory_system:
                return False
            
            # Check memory system health
            health = await self.memory_system.get_health_status()
            self.status.is_healthy = health.get("status") == "healthy"
            self.status.last_health_check = datetime.utcnow()
            
            return self.status.is_healthy
        except Exception as e:
            self.status.error_message = str(e)
            self.status.is_healthy = False
            logger.error(f"Memory health check failed: {e}")
            return False
    
    async def dispose(self):
        """Dispose memory system."""
        try:
            if self.memory_system:
                await self.memory_system.dispose()
            self.status.is_initialized = False
            self.status.is_running = False
            logger.info("Memory component disposed")
        except Exception as e:
            logger.error(f"Failed to dispose Memory component: {e}")


class ToolsComponent(SystemComponent):
    """Tools system component."""
    
    def __init__(self):
        super().__init__("Tools")
        self.tool_repository = None
    
    async def initialize(self) -> bool:
        """Initialize tools system."""
        try:
            from app.tools.unified_tool_repository import UnifiedToolRepository
            
            self.tool_repository = UnifiedToolRepository()
            await self.tool_repository.initialize()
            
            self.status.is_initialized = True
            logger.info("Tools component initialized")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to initialize Tools component: {e}")
            return False
    
    async def start(self) -> bool:
        """Start tools system."""
        try:
            if not self.status.is_initialized:
                return False
            
            self.status.is_running = True
            logger.info("Tools component started")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to start Tools component: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop tools system."""
        try:
            self.status.is_running = False
            logger.info("Tools component stopped")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to stop Tools component: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check tools system health."""
        try:
            if not self.tool_repository:
                return False
            
            # Check tools system health
            health = await self.tool_repository.get_health_status()
            self.status.is_healthy = health.get("status") == "healthy"
            self.status.last_health_check = datetime.utcnow()
            
            return self.status.is_healthy
        except Exception as e:
            self.status.error_message = str(e)
            self.status.is_healthy = False
            logger.error(f"Tools health check failed: {e}")
            return False
    
    async def dispose(self):
        """Dispose tools system."""
        try:
            if self.tool_repository:
                await self.tool_repository.dispose()
            self.status.is_initialized = False
            self.status.is_running = False
            logger.info("Tools component disposed")
        except Exception as e:
            logger.error(f"Failed to dispose Tools component: {e}")


class CommunicationComponent(SystemComponent):
    """Communication system component."""
    
    def __init__(self):
        super().__init__("Communication")
        self.communication_system = None
    
    async def initialize(self) -> bool:
        """Initialize communication system."""
        try:
            from app.communication.agent_communication_system import AgentCommunicationSystem
            
            self.communication_system = AgentCommunicationSystem()
            await self.communication_system.initialize()
            
            self.status.is_initialized = True
            logger.info("Communication component initialized")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to initialize Communication component: {e}")
            return False
    
    async def start(self) -> bool:
        """Start communication system."""
        try:
            if not self.status.is_initialized:
                return False
            
            self.status.is_running = True
            logger.info("Communication component started")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to start Communication component: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop communication system."""
        try:
            self.status.is_running = False
            logger.info("Communication component stopped")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to stop Communication component: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check communication system health."""
        try:
            if not self.communication_system:
                return False
            
            # Check communication system health
            health = await self.communication_system.get_health_status()
            self.status.is_healthy = health.get("status") == "healthy"
            self.status.last_health_check = datetime.utcnow()
            
            return self.status.is_healthy
        except Exception as e:
            self.status.error_message = str(e)
            self.status.is_healthy = False
            logger.error(f"Communication health check failed: {e}")
            return False
    
    async def dispose(self):
        """Dispose communication system."""
        try:
            if self.communication_system:
                await self.communication_system.dispose()
            self.status.is_initialized = False
            self.status.is_running = False
            logger.info("Communication component disposed")
        except Exception as e:
            logger.error(f"Failed to dispose Communication component: {e}")


class MonitoringComponent(SystemComponent):
    """Monitoring system component."""
    
    def __init__(self):
        super().__init__("Monitoring")
        self.monitoring_service = None
    
    async def initialize(self) -> bool:
        """Initialize monitoring system."""
        try:
            from app.core.monitoring import MonitoringService
            
            self.monitoring_service = MonitoringService()
            await self.monitoring_service.initialize()
            
            self.status.is_initialized = True
            logger.info("Monitoring component initialized")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to initialize Monitoring component: {e}")
            return False
    
    async def start(self) -> bool:
        """Start monitoring system."""
        try:
            if not self.status.is_initialized:
                return False
            
            await self.monitoring_service.start_monitoring()
            self.status.is_running = True
            logger.info("Monitoring component started")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to start Monitoring component: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop monitoring system."""
        try:
            if self.monitoring_service:
                self.monitoring_service.monitoring_enabled = False
            self.status.is_running = False
            logger.info("Monitoring component stopped")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to stop Monitoring component: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check monitoring system health."""
        try:
            if not self.monitoring_service:
                return False
            
            # Check monitoring system health
            health = await self.monitoring_service.get_health_status()
            self.status.is_healthy = health.get("status") == "healthy"
            self.status.last_health_check = datetime.utcnow()
            
            return self.status.is_healthy
        except Exception as e:
            self.status.error_message = str(e)
            self.status.is_healthy = False
            logger.error(f"Monitoring health check failed: {e}")
            return False
    
    async def dispose(self):
        """Dispose monitoring system."""
        try:
            if self.monitoring_service:
                await self.monitoring_service.dispose()
            self.status.is_initialized = False
            self.status.is_running = False
            logger.info("Monitoring component disposed")
        except Exception as e:
            logger.error(f"Failed to dispose Monitoring component: {e}")


class SecurityComponent(SystemComponent):
    """Security system component."""
    
    def __init__(self):
        super().__init__("Security")
        self.security_manager = None
    
    async def initialize(self) -> bool:
        """Initialize security system."""
        try:
            from app.core.security import SecurityManager
            
            self.security_manager = SecurityManager()
            
            self.status.is_initialized = True
            logger.info("Security component initialized")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to initialize Security component: {e}")
            return False
    
    async def start(self) -> bool:
        """Start security system."""
        try:
            if not self.status.is_initialized:
                return False
            
            self.status.is_running = True
            logger.info("Security component started")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to start Security component: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop security system."""
        try:
            self.status.is_running = False
            logger.info("Security component stopped")
            return True
        except Exception as e:
            self.status.error_message = str(e)
            logger.error(f"Failed to stop Security component: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check security system health."""
        try:
            if not self.security_manager:
                return False
            
            # Check security system health
            status = self.security_manager.get_security_status()
            self.status.is_healthy = status.get("blocked_ips", 0) < 1000  # Simple health check
            self.status.last_health_check = datetime.utcnow()
            
            return self.status.is_healthy
        except Exception as e:
            self.status.error_message = str(e)
            self.status.is_healthy = False
            logger.error(f"Security health check failed: {e}")
            return False
    
    async def dispose(self):
        """Dispose security system."""
        try:
            self.status.is_initialized = False
            self.status.is_running = False
            logger.info("Security component disposed")
        except Exception as e:
            logger.error(f"Failed to dispose Security component: {e}")


# Export all components
__all__ = [
    "ComponentStatus", "SystemComponent", "RAGComponent", "MemoryComponent",
    "ToolsComponent", "CommunicationComponent", "MonitoringComponent", "SecurityComponent"
]


