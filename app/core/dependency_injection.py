"""
Dependency injection system for the Agentic AI platform.

This module provides a clean dependency injection container that manages
all service dependencies and their lifecycle.
"""

from typing import Type, TypeVar, Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio


T = TypeVar('T')

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

# Get backend logger instance
_backend_logger = get_backend_logger()



class ServiceLifetime(Enum):
    """Service lifetime options."""
    SINGLETON = "singleton"
    SCOPED = "scoped"
    TRANSIENT = "transient"


@dataclass
class ServiceRegistration:
    """Service registration information."""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    dependencies: Optional[Dict[str, Type]] = None


class ServiceContainer:
    """Dependency injection container."""
    
    def __init__(self):
        self._services: Dict[Type, ServiceRegistration] = {}
        self._instances: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
        self._lock = asyncio.Lock()
    
    def register_singleton(self, service_type: Type, implementation_type: Optional[Type] = None, 
                         factory: Optional[Callable] = None, instance: Optional[Any] = None):
        """Register a singleton service."""
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
    
    def register_scoped(self, service_type: Type, implementation_type: Optional[Type] = None,
                       factory: Optional[Callable] = None):
        """Register a scoped service."""
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=ServiceLifetime.SCOPED
        )
    
    def register_transient(self, service_type: Type, implementation_type: Optional[Type] = None,
                          factory: Optional[Callable] = None):
        """Register a transient service."""
        self._services[service_type] = ServiceRegistration(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=ServiceLifetime.TRANSIENT
        )
    
    async def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance."""
        async with self._lock:
            if service_type not in self._services:
                raise ValueError(f"Service {service_type} not registered")
            
            registration = self._services[service_type]
            
            # Handle different lifetimes
            if registration.lifetime == ServiceLifetime.SINGLETON:
                return await self._get_singleton(registration)
            elif registration.lifetime == ServiceLifetime.SCOPED:
                return await self._get_scoped(registration)
            else:  # TRANSIENT
                return await self._create_instance(registration)
    
    async def _get_singleton(self, registration: ServiceRegistration) -> Any:
        """Get or create singleton instance."""
        if registration.instance is not None:
            return registration.instance
        
        if registration.service_type in self._instances:
            return self._instances[registration.service_type]
        
        instance = await self._create_instance(registration)
        self._instances[registration.service_type] = instance
        return instance
    
    async def _get_scoped(self, registration: ServiceRegistration) -> Any:
        """Get or create scoped instance."""
        if not self._current_scope:
            raise ValueError("No active scope for scoped service")
        
        if self._current_scope not in self._scoped_instances:
            self._scoped_instances[self._current_scope] = {}
        
        scope_instances = self._scoped_instances[self._current_scope]
        
        if registration.service_type in scope_instances:
            return scope_instances[registration.service_type]
        
        instance = await self._create_instance(registration)
        scope_instances[registration.service_type] = instance
        return instance
    
    async def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create a new instance of the service."""
        if registration.factory:
            return await self._call_factory(registration.factory)
        
        if registration.implementation_type:
            return await self._create_from_type(registration.implementation_type)
        
        raise ValueError(f"Cannot create instance for {registration.service_type}")
    
    async def _call_factory(self, factory: Callable) -> Any:
        """Call a factory function."""
        if asyncio.iscoroutinefunction(factory):
            return await factory()
        else:
            return factory()
    
    async def _create_from_type(self, implementation_type: Type) -> Any:
        """Create instance from type with dependency injection."""
        # Get constructor parameters
        import inspect
        sig = inspect.signature(implementation_type.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation != inspect.Parameter.empty:
                param_value = await self.get_service(param.annotation)
                params[param_name] = param_value
        
        return implementation_type(**params)
    
    def create_scope(self) -> 'ServiceScope':
        """Create a new service scope."""
        return ServiceScope(self)
    
    async def dispose(self):
        """Dispose all services."""
        async with self._lock:
            # Dispose singleton instances
            for instance in self._instances.values():
                if hasattr(instance, 'dispose'):
                    if asyncio.iscoroutinefunction(instance.dispose):
                        await instance.dispose()
                    else:
                        instance.dispose()
            
            # Dispose scoped instances
            for scope_instances in self._scoped_instances.values():
                for instance in scope_instances.values():
                    if hasattr(instance, 'dispose'):
                        if asyncio.iscoroutinefunction(instance.dispose):
                            await instance.dispose()
                        else:
                            instance.dispose()
            
            self._instances.clear()
            self._scoped_instances.clear()


class ServiceScope:
    """Service scope for managing scoped services."""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
        self.scope_id = f"scope_{id(self)}"
        self.container._current_scope = self.scope_id
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.dispose()
    
    async def dispose(self):
        """Dispose all scoped instances."""
        if self.scope_id in self.container._scoped_instances:
            scope_instances = self.container._scoped_instances[self.scope_id]
            for instance in scope_instances.values():
                if hasattr(instance, 'dispose'):
                    if asyncio.iscoroutinefunction(instance.dispose):
                        await instance.dispose()
                    else:
                        instance.dispose()
            
            del self.container._scoped_instances[self.scope_id]
        
        self.container._current_scope = None


class ServiceProvider:
    """Service provider for accessing services."""
    
    def __init__(self, container: ServiceContainer):
        self.container = container
    
    async def get_service(self, service_type: Type[T]) -> T:
        """Get a service instance."""
        return await self.container.get_service(service_type)
    
    def create_scope(self) -> ServiceScope:
        """Create a new service scope."""
        return self.container.create_scope()


# Service interfaces
class IDisposable(ABC):
    """Interface for disposable services."""
    
    @abstractmethod
    async def dispose(self):
        """Dispose the service."""
        pass


class IConfigurable(ABC):
    """Interface for configurable services."""
    
    @abstractmethod
    async def configure(self, config: Dict[str, Any]):
        """Configure the service."""
        pass


class IInitializable(ABC):
    """Interface for initializable services."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the service."""
        pass


# Service base classes
class BaseService(IDisposable, IConfigurable, IInitializable):
    """Base service class with common functionality."""
    
    def __init__(self):
        self._initialized = False
        self._disposed = False
        self._config = {}
    
    async def configure(self, config: Dict[str, Any]):
        """Configure the service."""
        self._config.update(config)
    
    async def initialize(self):
        """Initialize the service."""
        if self._initialized:
            return
        
        await self._on_initialize()
        self._initialized = True
    
    async def dispose(self):
        """Dispose the service."""
        if self._disposed:
            return
        
        await self._on_dispose()
        self._disposed = True
    
    async def _on_initialize(self):
        """Override in subclasses for initialization logic."""
        pass
    
    async def _on_dispose(self):
        """Override in subclasses for disposal logic."""
        pass


# Global service container
_service_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    """Get the global service container."""
    global _service_container
    if _service_container is None:
        _service_container = ServiceContainer()
    return _service_container


def get_service_provider() -> ServiceProvider:
    """Get the global service provider."""
    return ServiceProvider(get_service_container())


# Service registration helpers
def register_services(container: ServiceContainer):
    """Register all application services."""
    from app.core.monitoring import MonitoringService
    from app.core.security import SecurityManager
    from app.tools.unified_tool_repository import UnifiedToolRepository
    from app.rag.rag_system import RAGSystem
    from app.memory.memory_system import MemorySystem
    from app.communication.agent_communication_system import AgentCommunicationSystem
    
    # Core services
    container.register_singleton(MonitoringService, MonitoringService)
    container.register_singleton(SecurityManager, SecurityManager)
    container.register_singleton(UnifiedToolRepository, UnifiedToolRepository)
    container.register_singleton(RAGSystem, RAGSystem)
    container.register_singleton(MemorySystem, MemorySystem)
    container.register_singleton(AgentCommunicationSystem, AgentCommunicationSystem)
    
    # Register other services as needed
    _backend_logger.info(
        "All services registered successfully",
        LogCategory.SYSTEM_HEALTH,
        "app.core.dependency_injection"
    )


# Dependency injection decorators
def inject(service_type: Type[T]) -> T:
    """Dependency injection decorator."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            container = get_service_container()
            service = await container.get_service(service_type)
            return await func(service, *args, **kwargs)
        return wrapper
    return decorator


def scoped(func):
    """Scoped service decorator."""
    async def wrapper(*args, **kwargs):
        container = get_service_container()
        async with container.create_scope():
            return await func(*args, **kwargs)
    return wrapper


# Export all components
__all__ = [
    "ServiceLifetime", "ServiceRegistration", "ServiceContainer", "ServiceScope",
    "ServiceProvider", "IDisposable", "IConfigurable", "IInitializable",
    "BaseService", "get_service_container", "get_service_provider",
    "register_services", "inject", "scoped"
]


