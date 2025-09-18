"""
Unified System Orchestrator for Multi-Agent Architecture.

This module serves as the central orchestrator for the entire unified
multi-agent system, coordinating all components and providing a single
entry point for system initialization and management.

Features:
- Centralized system initialization
- Component lifecycle management
- Unified configuration management
- System health monitoring
- Graceful shutdown handling
- Inter-component coordination
"""

import asyncio
import signal
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field

# Import all unified system components
from app.rag.core.unified_rag_system import UnifiedRAGSystem, UnifiedRAGConfig
from app.rag.core.collection_based_kb_manager import CollectionBasedKBManager
from app.rag.core.agent_isolation_manager import AgentIsolationManager
from app.rag.core.unified_memory_system import UnifiedMemorySystem, UnifiedMemoryConfig
from app.rag.core.agent_memory_collections import AgentMemoryCollections
from app.tools.unified_tool_repository import UnifiedToolRepository
from app.communication.agent_communication_system import AgentCommunicationSystem, CommunicationConfig
from app.communication.knowledge_sharing_protocols import KnowledgeSharingProtocol
from app.communication.collaboration_manager import CollaborationManager
from app.optimization.performance_optimizer import PerformanceOptimizer, OptimizationConfig
from app.optimization.advanced_access_controls import AdvancedAccessController
from app.optimization.monitoring_analytics import MonitoringSystem

logger = structlog.get_logger(__name__)


class SystemConfig(BaseModel):
    """Unified system configuration."""
    # Component configurations
    rag_config: UnifiedRAGConfig = Field(default_factory=UnifiedRAGConfig)
    memory_config: UnifiedMemoryConfig = Field(default_factory=UnifiedMemoryConfig)
    communication_config: CommunicationConfig = Field(default_factory=CommunicationConfig)
    optimization_config: OptimizationConfig = Field(default_factory=OptimizationConfig)
    
    # System settings
    enable_monitoring: bool = Field(default=True, description="Enable system monitoring")
    enable_optimization: bool = Field(default=True, description="Enable performance optimization")
    enable_security: bool = Field(default=True, description="Enable advanced security")
    
    # Startup settings
    auto_initialize_components: bool = Field(default=True, description="Auto-initialize all components")
    graceful_shutdown_timeout: int = Field(default=30, description="Graceful shutdown timeout in seconds")


@dataclass
class SystemStatus:
    """System status information."""
    is_initialized: bool = False
    is_running: bool = False
    start_time: Optional[datetime] = None
    components_status: Dict[str, bool] = None
    health_score: float = 0.0
    last_health_check: Optional[datetime] = None


class UnifiedSystemOrchestrator:
    """
    Unified System Orchestrator.
    
    Central coordinator for the entire multi-agent system, managing
    component lifecycle, configuration, and inter-component communication.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize the system orchestrator."""
        self.config = config or SystemConfig()
        
        # System status
        self.status = SystemStatus(components_status={})
        
        # Core components (initialized in order)
        self.unified_rag: Optional[UnifiedRAGSystem] = None
        self.isolation_manager: Optional[AgentIsolationManager] = None
        self.memory_system: Optional[UnifiedMemorySystem] = None
        self.kb_manager: Optional[CollectionBasedKBManager] = None
        self.memory_collections: Optional[AgentMemoryCollections] = None
        self.tool_repository: Optional[UnifiedToolRepository] = None
        self.communication_system: Optional[AgentCommunicationSystem] = None
        self.knowledge_sharing: Optional[KnowledgeSharingProtocol] = None
        self.collaboration_manager: Optional[CollaborationManager] = None
        
        # Optimization and monitoring components
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.access_controller: Optional[AdvancedAccessController] = None
        self.monitoring_system: Optional[MonitoringSystem] = None
        
        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()
        
        logger.info("Unified system orchestrator created", config=self.config.dict())
    
    async def initialize(self) -> None:
        """Initialize the entire unified system."""
        try:
            if self.status.is_initialized:
                logger.warning("System already initialized")
                return
            
            logger.info("🚀 Initializing Unified Multi-Agent System...")
            self.status.start_time = datetime.utcnow()
            
            # Phase 1: Core RAG and Memory Systems
            logger.info("📊 Phase 1: Initializing Core Systems...")
            await self._initialize_core_systems()
            
            # Phase 2: Tool and Communication Systems
            logger.info("🔧 Phase 2: Initializing Tool and Communication Systems...")
            await self._initialize_tool_communication_systems()
            
            # Phase 3: Optimization and Monitoring
            if self.config.enable_optimization or self.config.enable_monitoring:
                logger.info("⚡ Phase 3: Initializing Optimization and Monitoring...")
                await self._initialize_optimization_monitoring()
            
            # Phase 4: Final system validation
            logger.info("✅ Phase 4: Final System Validation...")
            await self._validate_system_integrity()
            
            self.status.is_initialized = True
            self.status.is_running = True
            
            logger.info("🎉 Unified Multi-Agent System initialized successfully!")
            await self._log_system_summary()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize unified system: {str(e)}")
            await self._cleanup_partial_initialization()
            raise
    
    async def _initialize_core_systems(self) -> None:
        """Initialize core RAG and memory systems."""
        try:
            # 1. Unified RAG System
            logger.info("Initializing Unified RAG System...")
            self.unified_rag = UnifiedRAGSystem(self.config.rag_config)
            await self.unified_rag.initialize()
            self.status.components_status["unified_rag"] = True
            
            # 2. Agent Isolation Manager
            logger.info("Initializing Agent Isolation Manager...")
            self.isolation_manager = AgentIsolationManager(self.unified_rag)
            self.status.components_status["isolation_manager"] = True
            
            # 3. Unified Memory System
            logger.info("Initializing Unified Memory System...")
            self.memory_system = UnifiedMemorySystem(
                self.unified_rag,
                self.isolation_manager,
                self.config.memory_config
            )
            await self.memory_system.initialize()
            self.status.components_status["memory_system"] = True
            
            # 4. Collection-based Knowledge Base Manager
            logger.info("Initializing Knowledge Base Manager...")
            self.kb_manager = CollectionBasedKBManager(self.unified_rag)
            self.status.components_status["kb_manager"] = True
            
            # 5. Agent Memory Collections
            logger.info("Initializing Agent Memory Collections...")
            self.memory_collections = AgentMemoryCollections(
                self.unified_rag,
                self.memory_system,
                self.isolation_manager
            )
            self.status.components_status["memory_collections"] = True
            
            logger.info("✅ Core systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize core systems: {str(e)}")
            raise
    
    async def _initialize_tool_communication_systems(self) -> None:
        """Initialize tool and communication systems."""
        try:
            # 1. Unified Tool Repository
            logger.info("Initializing Unified Tool Repository...")
            self.tool_repository = UnifiedToolRepository(self.isolation_manager)
            await self.tool_repository.initialize()
            self.status.components_status["tool_repository"] = True
            
            # 2. Agent Communication System
            logger.info("Initializing Agent Communication System...")
            self.communication_system = AgentCommunicationSystem(
                self.isolation_manager,
                self.config.communication_config
            )
            await self.communication_system.initialize()
            self.status.components_status["communication_system"] = True
            
            # 3. Knowledge Sharing Protocol
            logger.info("Initializing Knowledge Sharing Protocol...")
            self.knowledge_sharing = KnowledgeSharingProtocol(
                self.unified_rag,
                self.kb_manager,
                self.isolation_manager,
                self.communication_system
            )
            self.status.components_status["knowledge_sharing"] = True
            
            # 4. Collaboration Manager
            logger.info("Initializing Collaboration Manager...")
            self.collaboration_manager = CollaborationManager(
                self.communication_system,
                self.isolation_manager
            )
            self.status.components_status["collaboration_manager"] = True
            
            logger.info("✅ Tool and communication systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tool and communication systems: {str(e)}")
            raise
    
    async def _initialize_optimization_monitoring(self) -> None:
        """Initialize optimization and monitoring systems."""
        try:
            # 1. Performance Optimizer
            if self.config.enable_optimization:
                logger.info("Initializing Performance Optimizer...")
                self.performance_optimizer = PerformanceOptimizer(
                    self.unified_rag,
                    self.memory_system,
                    self.tool_repository,
                    self.communication_system,
                    self.config.optimization_config
                )
                await self.performance_optimizer.initialize()
                self.status.components_status["performance_optimizer"] = True
            
            # 2. Advanced Access Controller
            if self.config.enable_security:
                logger.info("Initializing Advanced Access Controller...")
                self.access_controller = AdvancedAccessController(self.isolation_manager)
                self.status.components_status["access_controller"] = True
            
            # 3. Monitoring System
            if self.config.enable_monitoring:
                logger.info("Initializing Monitoring System...")
                self.monitoring_system = MonitoringSystem(
                    self.performance_optimizer,
                    self.access_controller,
                    self.unified_rag,
                    self.memory_system,
                    self.tool_repository,
                    self.communication_system
                )
                await self.monitoring_system.initialize()
                self.status.components_status["monitoring_system"] = True
            
            logger.info("✅ Optimization and monitoring systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimization and monitoring systems: {str(e)}")
            raise
    
    async def _validate_system_integrity(self) -> None:
        """Validate system integrity and component connectivity."""
        try:
            logger.info("Validating system integrity...")
            
            # Check all required components are initialized
            required_components = [
                "unified_rag", "isolation_manager", "memory_system",
                "kb_manager", "memory_collections", "tool_repository",
                "communication_system", "knowledge_sharing", "collaboration_manager"
            ]
            
            for component in required_components:
                if not self.status.components_status.get(component, False):
                    raise RuntimeError(f"Required component {component} not initialized")
            
            # Test component connectivity
            await self._test_component_connectivity()
            
            # Calculate initial health score
            self.status.health_score = await self._calculate_health_score()
            self.status.last_health_check = datetime.utcnow()
            
            logger.info("✅ System integrity validation passed")
            
        except Exception as e:
            logger.error(f"System integrity validation failed: {str(e)}")
            raise
    
    async def _test_component_connectivity(self) -> None:
        """Test connectivity between components."""
        try:
            # Test agent creation flow
            test_agent_id = "test_agent_system_validation"
            
            # Create agent isolation profile
            await self.isolation_manager.create_agent_isolation(test_agent_id)
            
            # Create agent memory
            await self.memory_system.create_agent_memory(test_agent_id)
            
            # Create agent tool profile
            await self.tool_repository.create_agent_profile(test_agent_id)
            
            # Register agent for communication
            await self.communication_system.register_agent(test_agent_id)
            
            # Cleanup test agent
            # Note: In a real implementation, we'd have cleanup methods
            
            logger.info("Component connectivity test passed")
            
        except Exception as e:
            logger.error(f"Component connectivity test failed: {str(e)}")
            raise
    
    async def _calculate_health_score(self) -> float:
        """Calculate overall system health score."""
        try:
            total_components = len(self.status.components_status)
            healthy_components = sum(1 for status in self.status.components_status.values() if status)
            
            if total_components == 0:
                return 0.0
            
            base_score = (healthy_components / total_components) * 100
            
            # Adjust based on performance metrics if available
            if self.performance_optimizer:
                try:
                    perf_report = self.performance_optimizer.get_performance_report()
                    if "current_metrics" in perf_report:
                        metrics = perf_report["current_metrics"]
                        # Factor in CPU and memory usage
                        cpu_factor = max(0, (100 - metrics.get("cpu_usage", 0)) / 100)
                        memory_factor = max(0, (100 - metrics.get("memory_usage", 0)) / 100)
                        base_score = base_score * 0.7 + (cpu_factor + memory_factor) * 15
                except Exception:
                    pass  # Use base score if performance metrics unavailable
            
            return min(100.0, max(0.0, base_score))
            
        except Exception as e:
            logger.error(f"Failed to calculate health score: {str(e)}")
            return 0.0
    
    async def _log_system_summary(self) -> None:
        """Log comprehensive system summary."""
        try:
            summary = {
                "🏗️ System Architecture": "Unified Multi-Agent System",
                "⏱️ Initialization Time": f"{(datetime.utcnow() - self.status.start_time).total_seconds():.2f}s",
                "🔧 Components Initialized": len([c for c in self.status.components_status.values() if c]),
                "💯 System Health Score": f"{self.status.health_score:.1f}%",
                "📊 RAG System": "✅ Unified with collection-based isolation",
                "🧠 Memory System": "✅ Unified with agent-specific collections",
                "🛠️ Tool Repository": "✅ Centralized with access controls",
                "💬 Communication": "✅ Multi-agent with knowledge sharing",
                "🤝 Collaboration": "✅ Task coordination and workflows",
                "⚡ Optimization": "✅ Performance tuning and monitoring" if self.config.enable_optimization else "❌ Disabled",
                "🔒 Security": "✅ Advanced access controls" if self.config.enable_security else "❌ Disabled",
                "📈 Monitoring": "✅ Comprehensive analytics" if self.config.enable_monitoring else "❌ Disabled"
            }
            
            logger.info("🎯 UNIFIED SYSTEM SUMMARY:")
            for key, value in summary.items():
                logger.info(f"   {key}: {value}")
            
        except Exception as e:
            logger.error(f"Failed to log system summary: {str(e)}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the entire system."""
        try:
            logger.info("🛑 Initiating graceful system shutdown...")
            
            self.status.is_running = False
            
            # Shutdown components in reverse order
            components_to_shutdown = [
                ("monitoring_system", self.monitoring_system),
                ("performance_optimizer", self.performance_optimizer),
                ("collaboration_manager", self.collaboration_manager),
                ("knowledge_sharing", self.knowledge_sharing),
                ("communication_system", self.communication_system),
                ("tool_repository", self.tool_repository),
                ("memory_collections", self.memory_collections),
                ("memory_system", self.memory_system),
                ("unified_rag", self.unified_rag)
            ]
            
            for component_name, component in components_to_shutdown:
                if component and hasattr(component, 'shutdown'):
                    try:
                        logger.info(f"Shutting down {component_name}...")
                        await component.shutdown()
                        self.status.components_status[component_name] = False
                    except Exception as e:
                        logger.error(f"Error shutting down {component_name}: {str(e)}")
            
            logger.info("✅ System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {str(e)}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating shutdown...")
                asyncio.create_task(self.shutdown())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
        except Exception as e:
            logger.error(f"Failed to setup signal handlers: {str(e)}")
    
    async def _cleanup_partial_initialization(self) -> None:
        """Cleanup after partial initialization failure."""
        try:
            logger.info("Cleaning up partial initialization...")
            await self.shutdown()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "is_initialized": self.status.is_initialized,
            "is_running": self.status.is_running,
            "start_time": self.status.start_time.isoformat() if self.status.start_time else None,
            "uptime_seconds": (datetime.utcnow() - self.status.start_time).total_seconds() if self.status.start_time else 0,
            "health_score": self.status.health_score,
            "last_health_check": self.status.last_health_check.isoformat() if self.status.last_health_check else None,
            "components_status": self.status.components_status,
            "config": self.config.dict()
        }


# Global system instance
_system_orchestrator: Optional[UnifiedSystemOrchestrator] = None


async def get_system_orchestrator(config: Optional[SystemConfig] = None) -> UnifiedSystemOrchestrator:
    """Get or create the global system orchestrator."""
    global _system_orchestrator
    
    if _system_orchestrator is None:
        _system_orchestrator = UnifiedSystemOrchestrator(config)
        await _system_orchestrator.initialize()
    
    return _system_orchestrator


async def shutdown_system() -> None:
    """Shutdown the global system orchestrator."""
    global _system_orchestrator
    
    if _system_orchestrator:
        await _system_orchestrator.shutdown()
        _system_orchestrator = None
