"""
Component manager for orchestrating system components.

This module provides a clean, modular way to manage system components
without the complexity of the monolithic orchestrator.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime
import asyncio

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory
from app.core.system_components import (
    SystemComponent, ComponentStatus, RAGComponent, MemoryComponent,
    ToolsComponent, CommunicationComponent, MonitoringComponent, SecurityComponent
)

_backend_logger = get_backend_logger()


@dataclass
class SystemHealth:
    """Overall system health status."""
    is_healthy: bool = False
    healthy_components: int = 0
    total_components: int = 0
    unhealthy_components: List[str] = None
    last_check: Optional[datetime] = None
    overall_score: float = 0.0


class ComponentManager:
    """Manages system components with proper dependency resolution."""
    
    def __init__(self):
        self.components: Dict[str, SystemComponent] = {}
        self.component_order: List[str] = []
        self.is_initialized = False
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Initialize core components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Core components (no dependencies)
        self.components["RAG"] = RAGComponent()
        self.components["Memory"] = MemoryComponent()
        self.components["Tools"] = ToolsComponent()
        self.components["Security"] = SecurityComponent()
        
        # Optional components
        self.components["Communication"] = CommunicationComponent()
        self.components["Monitoring"] = MonitoringComponent()
        
        # Set up dependencies
        self._setup_dependencies()
        
        # Calculate initialization order
        self._calculate_initialization_order()
    
    def _setup_dependencies(self):
        """Set up component dependencies."""
        # RAG has no dependencies
        # Memory has no dependencies
        # Tools has no dependencies
        # Security has no dependencies
        
        # Communication depends on RAG and Memory
        self.components["Communication"].add_dependency("RAG")
        self.components["Communication"].add_dependency("Memory")
        
        # Monitoring depends on all other components
        self.components["Monitoring"].add_dependency("RAG")
        self.components["Monitoring"].add_dependency("Memory")
        self.components["Monitoring"].add_dependency("Tools")
        self.components["Monitoring"].add_dependency("Security")
    
    def _calculate_initialization_order(self):
        """Calculate the order in which components should be initialized."""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(component_name: str):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {component_name}")
            if component_name in visited:
                return
            
            temp_visited.add(component_name)
            
            # Visit dependencies first
            component = self.components[component_name]
            for dep in component.dependencies:
                if dep in self.components:
                    visit(dep)
            
            temp_visited.remove(component_name)
            visited.add(component_name)
            order.append(component_name)
        
        # Visit all components
        for component_name in self.components:
            if component_name not in visited:
                visit(component_name)
        
        self.component_order = order
        _backend_logger.info(
            f"Component initialization order: {self.component_order}",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.component_manager"
        )

    async def initialize_all(self) -> bool:
        """Initialize all components in dependency order."""
        if self.is_initialized:
            return True

        _backend_logger.info(
            "Initializing all system components",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.component_manager"
        )

        try:
            # Initialize components in dependency order
            for component_name in self.component_order:
                component = self.components[component_name]
                _backend_logger.info(
                    f"Initializing component: {component_name}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )

                success = await component.initialize()
                if not success:
                    _backend_logger.error(
                        f"Failed to initialize component: {component_name}",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.component_manager"
                    )
                    return False

                _backend_logger.info(
                    f"Component {component_name} initialized successfully",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )

            self.is_initialized = True
            _backend_logger.info(
                "All components initialized successfully",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return True

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize components: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return False

    async def start_all(self) -> bool:
        """Start all components."""
        if not self.is_initialized:
            _backend_logger.error(
                "Components not initialized",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return False
        
        if self.is_running:
            return True

        _backend_logger.info(
            "Starting all system components",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.component_manager"
        )

        try:
            # Start components in dependency order
            for component_name in self.component_order:
                component = self.components[component_name]
                _backend_logger.info(
                    f"Starting component: {component_name}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )

                success = await component.start()
                if not success:
                    _backend_logger.error(
                        f"Failed to start component: {component_name}",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.component_manager"
                    )
                    return False

                _backend_logger.info(
                    f"Component {component_name} started successfully",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )

            self.is_running = True
            self.start_time = datetime.utcnow()
            _backend_logger.info(
                "All components started successfully",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return True

        except Exception as e:
            _backend_logger.error(
                f"Failed to start components: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return False
    
    async def stop_all(self) -> bool:
        """Stop all components in reverse order."""
        if not self.is_running:
            return True

        _backend_logger.info(
            "Stopping all system components",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.component_manager"
        )

        try:
            # Stop components in reverse dependency order
            for component_name in reversed(self.component_order):
                component = self.components[component_name]
                _backend_logger.info(
                    f"Stopping component: {component_name}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )

                success = await component.stop()
                if not success:
                    _backend_logger.warn(
                        f"Failed to stop component: {component_name}",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.component_manager"
                    )

                _backend_logger.info(
                    f"Component {component_name} stopped",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )

            self.is_running = False
            _backend_logger.info(
                "All components stopped",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return True

        except Exception as e:
            _backend_logger.error(
                f"Failed to stop components: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return False
    
    async def health_check_all(self) -> SystemHealth:
        """Check health of all components."""
        healthy_components = 0
        unhealthy_components = []
        
        for component_name, component in self.components.items():
            try:
                is_healthy = await component.health_check()
                if is_healthy:
                    healthy_components += 1
                else:
                    unhealthy_components.append(component_name)
            except Exception as e:
                _backend_logger.error(
                    f"Health check failed for {component_name}: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )
                unhealthy_components.append(component_name)
        
        total_components = len(self.components)
        is_healthy = len(unhealthy_components) == 0
        overall_score = healthy_components / total_components if total_components > 0 else 0.0
        
        return SystemHealth(
            is_healthy=is_healthy,
            healthy_components=healthy_components,
            total_components=total_components,
            unhealthy_components=unhealthy_components,
            last_check=datetime.utcnow(),
            overall_score=overall_score
        )
    
    async def dispose_all(self):
        """Dispose all components."""
        _backend_logger.info(
            "Disposing all system components",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.component_manager"
        )

        try:
            # Dispose components in reverse dependency order
            for component_name in reversed(self.component_order):
                component = self.components[component_name]
                _backend_logger.info(
                    f"Disposing component: {component_name}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )

                try:
                    await component.dispose()
                    _backend_logger.info(
                        f"Component {component_name} disposed",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.component_manager"
                    )
                except Exception as e:
                    _backend_logger.error(
                        f"Failed to dispose component {component_name}: {e}",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.component_manager"
                    )

            self.is_initialized = False
            self.is_running = False
            _backend_logger.info(
                "All components disposed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to dispose components: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
    
    def get_component(self, name: str) -> Optional[SystemComponent]:
        """Get a specific component."""
        return self.components.get(name)
    
    def get_component_status(self, name: str) -> Optional[ComponentStatus]:
        """Get status of a specific component."""
        component = self.get_component(name)
        return component.status if component else None
    
    def get_all_status(self) -> Dict[str, ComponentStatus]:
        """Get status of all components."""
        return {name: component.status for name, component in self.components.items()}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        health = asyncio.create_task(self.health_check_all())
        
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "component_count": len(self.components),
            "component_order": self.component_order,
            "components": {
                name: {
                    "is_initialized": component.status.is_initialized,
                    "is_running": component.status.is_running,
                    "is_healthy": component.status.is_healthy,
                    "last_health_check": component.status.last_health_check.isoformat() if component.status.last_health_check else None,
                    "error_message": component.status.error_message
                }
                for name, component in self.components.items()
            }
        }
    
    async def restart_component(self, name: str) -> bool:
        """Restart a specific component."""
        component = self.get_component(name)
        if not component:
            _backend_logger.error(
                f"Component {name} not found",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return False

        _backend_logger.info(
            f"Restarting component: {name}",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.component_manager"
        )

        try:
            # Stop component
            await component.stop()

            # Start component
            success = await component.start()
            if success:
                _backend_logger.info(
                    f"Component {name} restarted successfully",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )
            else:
                _backend_logger.error(
                    f"Failed to restart component {name}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.component_manager"
                )

            return success

        except Exception as e:
            _backend_logger.error(
                f"Failed to restart component {name}: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return False

    async def enable_component(self, name: str) -> bool:
        """Enable a component (if it was disabled)."""
        component = self.get_component(name)
        if not component:
            _backend_logger.error(
                f"Component {name} not found",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return False
        
        if not component.status.is_initialized:
            success = await component.initialize()
            if not success:
                return False
        
        if not component.status.is_running:
            success = await component.start()
            if not success:
                return False

        _backend_logger.info(
            f"Component {name} enabled",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.component_manager"
        )
        return True

    async def disable_component(self, name: str) -> bool:
        """Disable a component."""
        component = self.get_component(name)
        if not component:
            _backend_logger.error(
                f"Component {name} not found",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return False

        _backend_logger.info(
            f"Disabling component: {name}",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.component_manager"
        )

        try:
            await component.stop()
            await component.dispose()
            _backend_logger.info(
                f"Component {name} disabled",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return True
        except Exception as e:
            _backend_logger.error(
                f"Failed to disable component {name}: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.component_manager"
            )
            return False


# Global component manager instance
_component_manager: Optional[ComponentManager] = None


def get_component_manager() -> ComponentManager:
    """Get the global component manager."""
    global _component_manager
    if _component_manager is None:
        _component_manager = ComponentManager()
    return _component_manager


# Convenience functions
async def initialize_system() -> bool:
    """Initialize the entire system."""
    manager = get_component_manager()
    return await manager.initialize_all()


async def start_system() -> bool:
    """Start the entire system."""
    manager = get_component_manager()
    return await manager.start_all()


async def stop_system() -> bool:
    """Stop the entire system."""
    manager = get_component_manager()
    return await manager.stop_all()


async def get_system_health() -> SystemHealth:
    """Get system health status."""
    manager = get_component_manager()
    return await manager.health_check_all()


def get_system_status() -> Dict[str, Any]:
    """Get system status."""
    manager = get_component_manager()
    return manager.get_system_status()


# Export all components
__all__ = [
    "SystemHealth", "ComponentManager", "get_component_manager",
    "initialize_system", "start_system", "stop_system", "get_system_health", "get_system_status"
]


