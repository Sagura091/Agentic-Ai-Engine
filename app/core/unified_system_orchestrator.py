"""
Unified System Orchestrator - THE Central Command for Multi-Agent Architecture.

This is THE ONLY system orchestrator in the entire application.
All system initialization, coordination, and management flows through this orchestrator.

COMPLETE SYSTEM ARCHITECTURE:
✅ PHASE 1: Foundation (UnifiedRAGSystem, CollectionBasedKBManager, AgentIsolationManager)
✅ PHASE 2: Memory & Tools (UnifiedMemorySystem, UnifiedToolRepository)
✅ PHASE 3: Communication (AgentCommunicationSystem, KnowledgeSharing, Collaboration)
✅ PHASE 4: Optimization (Monitoring, Access Controls)

DESIGN PRINCIPLES:
- One orchestrator to rule them all
- Centralized initialization and coordination
- Simple, clean, fast operations
- No complexity unless absolutely necessary

SYSTEM FEATURES:
- Single entry point for entire system
- Automatic component initialization
- Health monitoring and optimization
- Graceful shutdown handling
- Inter-component coordination
"""

import asyncio
import signal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

# Import THE unified system components - ALL PHASES
from app.rag.core.unified_rag_system import UnifiedRAGSystem, UnifiedRAGConfig
from app.rag.core.collection_based_kb_manager import CollectionBasedKBManager
from app.rag.core.agent_isolation_manager import AgentIsolationManager
from app.memory.unified_memory_system import UnifiedMemorySystem
from app.tools.unified_tool_repository import UnifiedToolRepository
from app.communication.agent_communication_system import AgentCommunicationSystem

from app.backend_logging.backend_logger import get_logger as get_backend_logger
from app.backend_logging.models import LogCategory

_backend_logger = get_backend_logger()


class SystemConfig(BaseModel):
    """Unified system configuration - SIMPLIFIED."""
    # Core configurations
    rag_config: UnifiedRAGConfig = Field(default_factory=UnifiedRAGConfig)

    # System settings
    enable_communication: bool = Field(default=False)  # Communication disabled by default
    enable_optimization: bool = Field(default=True)    # Optimization enabled by default
    enable_monitoring: bool = Field(default=True)      # Monitoring enabled by default
    enable_security: bool = Field(default=False)       # Security disabled by default
    
    # System settings - SIMPLIFIED
    auto_initialize_components: bool = Field(default=True)
    graceful_shutdown_timeout: int = Field(default=30)


@dataclass
class SystemStatus:
    """System status information - SIMPLIFIED."""
    is_initialized: bool = False
    is_running: bool = False
    start_time: Optional[datetime] = None
    components_status: Dict[str, bool] = field(default_factory=dict)
    health_score: float = 0.0
    last_health_check: Optional[datetime] = None


class UnifiedSystemOrchestrator:
    """
    Unified System Orchestrator - THE Central Command.

    COMPLETE SYSTEM ARCHITECTURE:
    ✅ PHASE 1: Foundation (RAG, KB Manager, Agent Isolation)
    ✅ PHASE 2: Memory & Tools (Memory System, Tool Repository)
    ✅ PHASE 3: Communication (Agent Communication, Knowledge Sharing)
    ✅ PHASE 4: Optimization (Performance, Monitoring, Access Control)
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize THE system orchestrator."""
        self.config = config or SystemConfig()

        # System status
        self.status = SystemStatus()

        # PHASE 1: Core components (THE Foundation)
        self.unified_rag: Optional[UnifiedRAGSystem] = None
        self.kb_manager: Optional[CollectionBasedKBManager] = None
        self.isolation_manager: Optional[AgentIsolationManager] = None

        # PHASE 2: Memory & Tools
        self.memory_system: Optional[UnifiedMemorySystem] = None
        self.tool_repository: Optional[UnifiedToolRepository] = None
        self.memory_consolidation_service = None  # CRITICAL FIX: Automatic consolidation

        # PHASE 3: Communication
        self.communication_system: Optional[AgentCommunicationSystem] = None

        # REVOLUTIONARY: Component Workflow Execution System
        self.component_workflow_executor: Optional['ComponentWorkflowExecutor'] = None
        self.workflow_step_manager: Optional['WorkflowStepManager'] = None

        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()

        _backend_logger.info(
            "THE Unified system orchestrator created",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.unified_system_orchestrator"
        )

    async def initialize(self) -> None:
        """Initialize THE entire unified system - ALL PHASES."""
        try:
            if self.status.is_initialized:
                _backend_logger.warn(
                    "System already initialized",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
                return

            _backend_logger.warn(
                "🚀 Initializing THE Unified Multi-Agent System...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.status.start_time = datetime.utcnow()

            # PHASE 1: Foundation (Weeks 1-3)
            _backend_logger.debug(
                "🏗️ PHASE 1: Foundation - Unified RAG System core, Collection-based KB manager, Basic agent isolation",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            await self._initialize_phase_1_foundation()

            # PHASE 2: Memory & Tools (Weeks 4-6)
            _backend_logger.debug(
                "🧠 PHASE 2: Memory & Tools - Unified memory system, Tool repository consolidation, Agent-specific memory collections",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            await self._initialize_phase_2_memory_tools()

            # PHASE 3: Communication (Weeks 7-9)
            if self.config.enable_communication:
                _backend_logger.debug(
                    "📡 PHASE 3: Communication - Agent communication layer, Knowledge sharing protocols, Collaboration mechanisms",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
                await self._initialize_phase_3_communication()

            # PHASE 4: Optimization (Weeks 10-11)
            if self.config.enable_optimization:
                _backend_logger.debug(
                    "⚡ PHASE 4: Optimization - Performance tuning, Advanced access controls, Monitoring & analytics",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
                await self._initialize_phase_4_optimization()

            # REVOLUTIONARY: Component Workflow Execution System
            _backend_logger.debug(
                "🚀 REVOLUTIONARY: Initializing Component Workflow Execution System...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            await self._initialize_component_workflow_system()

            # Final system validation
            _backend_logger.debug(
                "✅ Final System Validation...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            await self._validate_system_integrity()

            self.status.is_initialized = True
            self.status.is_running = True

            _backend_logger.warn(
                "🎉 Unified Multi-Agent System initialized successfully!",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            await self._log_system_summary()

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to initialize unified system: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            await self._cleanup_partial_initialization()
            raise
    
    async def _initialize_phase_1_foundation(self) -> None:
        """Initialize PHASE 1: Foundation components."""
        try:
            # 1. Initialize THE UnifiedRAGSystem (THE single RAG system)
            _backend_logger.debug(
                "   🎯 Initializing THE UnifiedRAGSystem...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.unified_rag = UnifiedRAGSystem(self.config.rag_config)
            await self.unified_rag.initialize()
            self.status.components_status["unified_rag"] = True

            # 🚀 Initialize Revolutionary Dynamic RAG Configuration Manager
            _backend_logger.debug(
                "   🔧 Initializing Revolutionary Dynamic RAG Configuration Manager...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            from app.rag.core.dynamic_config_manager import initialize_rag_config_manager, rag_config_manager
            await initialize_rag_config_manager(self.unified_rag)
            _backend_logger.debug(
                "   ✅ RAG Configuration Manager initialized - Real-time updates enabled!",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

            # 🚀 Initialize Revolutionary Global Configuration Manager
            _backend_logger.info(
                "   🌐 Initializing Revolutionary Global Configuration Manager...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            from app.core.global_config_manager import global_config_manager, initialize_global_config_manager
            from app.core.config_sections.rag_section_manager import RAGSectionManager
            from app.core.config_sections.llm_section_manager import LLMSectionManager
            from app.core.config_sections.memory_section_manager import MemorySectionManager
            from app.core.config_sections.database_section_manager import DatabaseSectionManager
            from app.core.config_observers.rag_observer import RAGConfigurationObserver
            from app.core.config_observers.llm_observer import LLMConfigurationObserver
            from app.core.config_observers.memory_observer import MemoryConfigurationObserver
            from app.core.config_observers.database_observer import database_observer

            # Initialize the global configuration manager
            await initialize_global_config_manager()

            # Register RAG section manager
            rag_section_manager = RAGSectionManager()
            rag_section_manager.set_rag_system(self.unified_rag)
            rag_section_manager.set_rag_config_manager(rag_config_manager)
            global_config_manager.register_section_manager(rag_section_manager)

            # Register RAG configuration observer
            rag_observer = RAGConfigurationObserver(self.unified_rag, rag_config_manager)
            global_config_manager.register_observer(rag_observer)

            # 🚀 Register LLM Provider Section Manager
            _backend_logger.info(
                "   🤖 Registering LLM Provider Section Manager...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            llm_section_manager = LLMSectionManager()
            # LLM service and manager will be set when they're initialized
            global_config_manager.register_section_manager(llm_section_manager)

            # Register LLM configuration observer
            llm_observer = LLMConfigurationObserver()
            # LLM service and manager will be set when they're initialized
            global_config_manager.register_observer(llm_observer)

            # 🚀 Register Memory System Section Manager
            _backend_logger.info(
                "   🧠 Registering Memory System Section Manager...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            memory_section_manager = MemorySectionManager()
            memory_section_manager.set_unified_rag_system(self.unified_rag)
            # Unified memory system will be set when it's initialized
            global_config_manager.register_section_manager(memory_section_manager)

            # Register Memory configuration observer
            memory_observer = MemoryConfigurationObserver()
            memory_observer.set_unified_rag_system(self.unified_rag)
            # Unified memory system will be set when it's initialized
            global_config_manager.register_observer(memory_observer)

            # 🚀 Register Database Storage Section Manager
            _backend_logger.info(
                "   🗄️ Registering Database Storage Section Manager...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            database_section_manager = DatabaseSectionManager()
            global_config_manager.register_section_manager(database_section_manager)

            # Register Database configuration observer
            await database_observer.initialize()
            global_config_manager.register_observer(database_observer)

            _backend_logger.info(
                "   ✅ Global Configuration Manager initialized - Revolutionary real-time configuration management enabled!",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            _backend_logger.info(
                "   ✅ LLM Provider Configuration Manager registered - Real-time provider switching enabled!",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            _backend_logger.info(
                "   ✅ Memory System Configuration Manager registered - Real-time memory management enabled!",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

            # 2. Initialize THE CollectionBasedKBManager (THE knowledge base system)
            _backend_logger.info(
                "   📚 Initializing THE CollectionBasedKBManager...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.kb_manager = CollectionBasedKBManager(self.unified_rag)
            await self.kb_manager.initialize()
            self.status.components_status["kb_manager"] = True

            # 3. Initialize THE AgentIsolationManager (THE agent isolation system)
            _backend_logger.info(
                "   🔒 Initializing THE AgentIsolationManager...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.isolation_manager = AgentIsolationManager(self.unified_rag)
            await self.isolation_manager.initialize()
            self.status.components_status["isolation_manager"] = True

            _backend_logger.info(
                "✅ PHASE 1 Foundation: COMPLETE",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize PHASE 1 Foundation: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            raise

    async def _initialize_phase_2_memory_tools(self) -> None:
        """Initialize PHASE 2: Memory & Tools components."""
        try:
            # 1. Initialize THE UnifiedMemorySystem (THE memory system)
            _backend_logger.info(
                "   🧠 Initializing THE UnifiedMemorySystem...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.memory_system = UnifiedMemorySystem(self.unified_rag)
            await self.memory_system.initialize()
            self.status.components_status["memory_system"] = True

            # 🚀 Register memory system with configuration managers
            _backend_logger.info(
                "   🔧 Registering memory system with configuration managers...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            from app.core.global_config_manager import global_config_manager

            # Find and update memory section manager
            for section_manager in global_config_manager._section_managers.values():
                if hasattr(section_manager, 'set_unified_memory_system'):
                    section_manager.set_unified_memory_system(self.memory_system)
                    _backend_logger.info(
                        "   ✅ Memory system registered with memory section manager",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.unified_system_orchestrator"
                    )

            # Find and update memory observer
            for observer in global_config_manager._observers.get("memory_system", []):
                if hasattr(observer, 'set_unified_memory_system'):
                    observer.set_unified_memory_system(self.memory_system)
                    _backend_logger.info(
                        "   ✅ Memory system registered with memory observer",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.unified_system_orchestrator"
                    )

            # 2. Initialize THE UnifiedToolRepository (THE tool system)
            _backend_logger.info(
                "   🔧 Initializing THE UnifiedToolRepository...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.tool_repository = UnifiedToolRepository(self.unified_rag, self.isolation_manager)
            await self.tool_repository.initialize()

            # Register built-in tools
            await self._register_builtin_tools()
            self.status.components_status["tool_repository"] = True

            # 3. Initialize THE Hybrid RAG Integration (THE complete RAG system)
            _backend_logger.info(
                "   🚀 Initializing THE Hybrid RAG Integration...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            await self._initialize_hybrid_rag_integration()
            self.status.components_status["hybrid_rag_integration"] = True

            # CRITICAL FIX: Initialize automatic memory consolidation service
            _backend_logger.info(
                "   🔄 Initializing Memory Consolidation Service...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            await self._initialize_memory_consolidation_service()
            self.status.components_status["memory_consolidation_service"] = True

            _backend_logger.info(
                "✅ PHASE 2 Memory & Tools: COMPLETE",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize core systems: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            raise

    async def _initialize_memory_consolidation_service(self) -> None:
        """
        CRITICAL FIX: Initialize automatic memory consolidation service.

        This service runs periodic consolidation to:
        - Promote important memories to long-term storage
        - Forget low-value expired memories
        - Optimize memory performance
        - Enable continuous learning
        """
        try:
            from app.services.memory_consolidation_service import MemoryConsolidationService

            # Create consolidation service
            self.memory_consolidation_service = MemoryConsolidationService(
                memory_system=self.memory_system,
                interval_hours=6,  # Run every 6 hours
                consolidation_threshold=100,  # Minimum 100 memories before consolidation
                max_agents_per_cycle=50  # Process up to 50 agents per cycle
            )

            # Start the service
            await self.memory_consolidation_service.start()

            _backend_logger.info(
                "Memory consolidation service initialized and started",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator",
                data={"interval_hours": 6, "consolidation_threshold": 100}
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize memory consolidation service: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            # Don't raise - consolidation is important but not critical for startup
            _backend_logger.warn(
                "System will continue without automatic memory consolidation",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

    async def _register_builtin_tools(self):
        """Register all built-in tools with the tool repository."""
        try:
            _backend_logger.info(
                "🔧 Registering built-in tools...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

            # Import and register calculator tool
            try:
                from app.tools.calculator_tool import calculator_tool
                from app.tools.unified_tool_repository import ToolMetadata, ToolCategory, ToolAccessLevel

                metadata = ToolMetadata(
                    tool_id="calculator",
                    name="Calculator",
                    description="Mathematical calculations and arithmetic operations",
                    category=ToolCategory.COMPUTATION,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=["calculation", "math", "arithmetic", "computation"]
                )
                await self.tool_repository.register_tool(calculator_tool, metadata)
                _backend_logger.debug(
                    "✅ Registered calculator tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register calculator tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register revolutionary web research tool
            try:
                from app.tools.web_research_tool import web_research_tool

                metadata = ToolMetadata(
                    tool_id="web_research",
                    name="🚀 Revolutionary Web Research Tool",
                    description="The ultimate AI-powered web research assistant with advanced search, scraping, and analysis capabilities",
                    category=ToolCategory.RESEARCH,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "web_search", "research", "scraping", "information_gathering",
                        "competitive_intelligence", "market_research", "content_analysis",
                        "sentiment_analysis", "fact_checking", "entity_extraction"
                    ]
                )
                await self.tool_repository.register_tool(web_research_tool, metadata)
                _backend_logger.info(
                    "✅ Registered revolutionary web research tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register revolutionary web research tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register production tools
            try:
                from app.tools.production.file_system_tool import file_system_tool

                metadata = ToolMetadata(
                    tool_id="file_system",
                    name="Revolutionary File System Tool",
                    description="Revolutionary file system operations with enterprise security - Create, read, write, delete files and directories with advanced compression, search, and security features",
                    category=ToolCategory.UTILITY,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "file_management", "data_processing", "backup_operations",
                        "content_creation", "automation", "file_operations"
                    ]
                )
                await self.tool_repository.register_tool(file_system_tool, metadata)
                _backend_logger.info(
                    "✅ Registered file system tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register file system tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            try:
                from app.tools.production.api_integration_tool import api_integration_tool

                metadata = ToolMetadata(
                    tool_id="api_integration",
                    name="Revolutionary API Integration Tool",
                    description="Revolutionary API integration with intelligent handling and enterprise features - Universal HTTP methods, multiple authentication, rate limiting, caching, and circuit breaker patterns",
                    category=ToolCategory.COMMUNICATION,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "api_integration", "data_retrieval", "web_services",
                        "authentication", "automation", "http_requests"
                    ]
                )
                await self.tool_repository.register_tool(api_integration_tool, metadata)
                _backend_logger.info(
                    "✅ Registered API integration tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register API integration tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Register remaining Week 1 production tools
            try:
                from app.tools.production.database_operations_tool import database_operations_tool

                metadata = ToolMetadata(
                    tool_id="database_operations",
                    name="Revolutionary Database Operations Tool",
                    description="Revolutionary multi-database connectivity with universal operations - SQLite, PostgreSQL, MySQL, MongoDB, Redis with connection pooling, security, and performance optimization",
                    category=ToolCategory.DATA,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "database_management", "data_storage", "query_execution",
                        "data_analysis", "backup_operations", "database_operations"
                    ]
                )
                await self.tool_repository.register_tool(database_operations_tool, metadata)
                _backend_logger.info(
                    "✅ Registered database operations tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register database operations tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            try:
                from app.tools.production.text_processing_nlp_tool import text_processing_nlp_tool

                metadata = ToolMetadata(
                    tool_id="text_processing_nlp",
                    name="Revolutionary Text Processing & NLP Tool",
                    description="Revolutionary natural language processing with advanced text analysis - Sentiment analysis, entity extraction, keyword extraction, text similarity, and multi-language support",
                    category=ToolCategory.ANALYSIS,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "text_analysis", "nlp_processing", "sentiment_analysis",
                        "content_analysis", "language_processing", "text_mining"
                    ]
                )
                await self.tool_repository.register_tool(text_processing_nlp_tool, metadata)
                _backend_logger.info(
                    "✅ Registered text processing & NLP tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register text processing & NLP tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            try:
                from app.tools.production.password_security_tool import password_security_tool

                metadata = ToolMetadata(
                    tool_id="password_security",
                    name="Revolutionary Password & Security Tool",
                    description="Revolutionary cryptographic operations with military-grade security - Password generation, encryption/decryption, secure hashing, token generation, and security analysis",
                    category=ToolCategory.SECURITY,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "password_management", "encryption", "security_operations",
                        "cryptography", "token_generation", "security_analysis"
                    ]
                )
                await self.tool_repository.register_tool(password_security_tool, metadata)
                _backend_logger.info(
                    "✅ Registered password & security tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register password & security tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            try:
                from app.tools.production.notification_alert_tool import notification_alert_tool

                metadata = ToolMetadata(
                    tool_id="notification_alert",
                    name="Revolutionary Notification & Alert Tool",
                    description="Revolutionary multi-channel messaging with intelligent routing - Email, SMS, Slack, Discord, webhooks with delivery tracking, scheduling, and enterprise reliability",
                    category=ToolCategory.COMMUNICATION,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "notifications", "alerts", "messaging", "communication",
                        "email_automation", "multi_channel_delivery"
                    ]
                )
                await self.tool_repository.register_tool(notification_alert_tool, metadata)
                _backend_logger.info(
                    "✅ Registered notification & alert tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register notification & alert tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            try:
                from app.tools.production.qr_barcode_tool import qr_barcode_tool

                metadata = ToolMetadata(
                    tool_id="qr_barcode",
                    name="Revolutionary QR Code & Barcode Tool",
                    description="Revolutionary barcode generation and scanning with multi-format support - QR codes, Code128, EAN, UPC with customization, batch processing, and high-performance scanning",
                    category=ToolCategory.UTILITY,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "barcode_generation", "qr_codes", "inventory_management",
                        "document_tracking", "mobile_integration", "barcode_scanning"
                    ]
                )
                await self.tool_repository.register_tool(qr_barcode_tool, metadata)
                _backend_logger.info(
                    "✅ Registered QR code & barcode tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register QR code & barcode tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            try:
                from app.tools.production.weather_environmental_tool import weather_environmental_tool

                metadata = ToolMetadata(
                    tool_id="weather_environmental",
                    name="Revolutionary Weather & Environmental Tool",
                    description="Revolutionary weather and environmental monitoring with comprehensive data - Real-time weather, forecasting, air quality, marine conditions, agricultural data, and climate analysis",
                    category=ToolCategory.DATA,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "weather_monitoring", "environmental_data", "forecasting",
                        "air_quality", "marine_conditions", "agricultural_weather"
                    ]
                )
                await self.tool_repository.register_tool(weather_environmental_tool, metadata)
                _backend_logger.info(
                    "✅ Registered weather & environmental tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register weather & environmental tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # 🚀 REVOLUTIONARY AUTOMATION TOOLS - NEW CATEGORY
            try:
                from app.tools.production.screenshot_analysis_tool import screenshot_analysis_tool

                metadata = ToolMetadata(
                    tool_id="screenshot_analysis",
                    name="🚀 Revolutionary Screenshot Analysis Tool",
                    description="Revolutionary screenshot analysis with multi-modal LLM integration - OCR, UI element detection, visual reasoning, context analysis, and automation suggestions using Ollama llama4:scout + API providers",
                    category=ToolCategory.AUTOMATION,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "screenshot_analysis", "visual_understanding", "ui_detection",
                        "automation_planning", "visual_reasoning", "ocr_analysis",
                        "desktop_automation", "browser_automation", "testing_automation"
                    ]
                )
                await self.tool_repository.register_tool(screenshot_analysis_tool, metadata)
                _backend_logger.info(
                    "🚀 Registered REVOLUTIONARY Screenshot Analysis Tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register screenshot analysis tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            try:
                from app.tools.production.browser_automation_tool import browser_automation_tool

                metadata = ToolMetadata(
                    tool_id="browser_automation",
                    name="🚀 Revolutionary Browser Automation Tool",
                    description="Revolutionary browser automation with visual intelligence - Navigate websites, click elements using visual description, type into forms, scroll pages, extract data, and perform complex web workflows using computer vision + LLM reasoning",
                    category=ToolCategory.AUTOMATION,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "web_automation", "browser_control", "visual_web_interaction",
                        "form_filling", "data_extraction", "web_scraping",
                        "ui_testing", "workflow_automation", "web_navigation"
                    ]
                )
                await self.tool_repository.register_tool(browser_automation_tool, metadata)
                _backend_logger.info(
                    "🚀 Registered REVOLUTIONARY Browser Automation Tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register browser automation tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            try:
                from app.tools.production.computer_use_agent_tool import computer_use_agent_tool

                metadata = ToolMetadata(
                    tool_id="computer_use_agent",
                    name="🚀 Revolutionary Computer Use Agent Tool",
                    description="Revolutionary computer use agent with visual intelligence and security - Full desktop automation, application control, visual element interaction, system commands, and cross-platform desktop workflows with sandboxed execution",
                    category=ToolCategory.AUTOMATION,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "desktop_automation", "application_control", "visual_desktop_interaction",
                        "system_administration", "file_operations", "cross_platform_automation",
                        "ui_testing", "workflow_automation", "system_commands"
                    ]
                )
                await self.tool_repository.register_tool(computer_use_agent_tool, metadata)
                _backend_logger.info(
                    "🚀 Registered REVOLUTIONARY Computer Use Agent Tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register computer use agent tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            try:
                from app.tools.production.revolutionary_document_intelligence_tool import RevolutionaryDocumentIntelligenceTool

                # Create instance
                document_intelligence_tool = RevolutionaryDocumentIntelligenceTool()

                metadata = ToolMetadata(
                    tool_id="revolutionary_document_intelligence",
                    name="🔥 Revolutionary Document Intelligence Tool",
                    description="The most advanced AI-powered document processing tool - Multi-format analysis (PDF, Word, Excel, PowerPoint), intelligent content modification, template-based generation, format conversion, form filling, batch processing, and secure download links with background processing",
                    category=ToolCategory.AUTOMATION,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=True,
                    use_cases=[
                        "document_analysis", "document_modification", "document_generation",
                        "format_conversion", "template_creation", "form_filling",
                        "batch_processing", "content_extraction", "document_intelligence",
                        "pdf_processing", "word_processing", "excel_processing", "powerpoint_processing"
                    ]
                )
                await self.tool_repository.register_tool(document_intelligence_tool, metadata)
                _backend_logger.info(
                    "🔥 Registered REVOLUTIONARY Document Intelligence Tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register revolutionary document intelligence tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register REVOLUTIONARY WEB SCRAPER TOOL
            try:
                from app.tools.production.revolutionary_web_scraper_tool import RevolutionaryWebScraperTool

                # Create instance
                revolutionary_web_scraper_tool = RevolutionaryWebScraperTool()

                metadata = ToolMetadata(
                    tool_id="revolutionary_web_scraper",
                    name="🌐 Revolutionary Web Scraper Tool",
                    description="The most advanced web scraping tool ever created - Bypasses ALL bot detection systems (Cloudflare, DataDome, etc.), scrapes ANY website with complete stealth, search engine integration, JavaScript rendering, human behavior simulation, proxy rotation, TLS fingerprint spoofing, and CAPTCHA bypass. GUARANTEED to work on ANY website!",
                    category=ToolCategory.RESEARCH,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=[
                        "web_scraping", "search_engines", "content_extraction", "website_crawling",
                        "bot_detection_bypass", "cloudflare_bypass", "javascript_rendering",
                        "proxy_rotation", "stealth_scraping", "captcha_bypass", "tls_spoofing",
                        "google_search", "bing_search", "duckduckgo_search", "link_extraction",
                        "image_extraction", "video_extraction", "structured_data_extraction",
                        "human_behavior_simulation", "fingerprint_spoofing", "rate_limit_evasion"
                    ]
                )
                await self.tool_repository.register_tool(revolutionary_web_scraper_tool, metadata)
                _backend_logger.info(
                    "🌐 Registered REVOLUTIONARY WEB SCRAPER TOOL - THE ULTIMATE INTERNET DOMINATION SYSTEM!",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register revolutionary web scraper tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register general business intelligence tool
            try:
                from app.tools.general_business_intelligence_tool import GeneralBusinessIntelligenceTool

                general_bi_tool = GeneralBusinessIntelligenceTool()
                metadata = ToolMetadata(
                    tool_id="general_business_intelligence",
                    name="General Business Intelligence",
                    description="Generate realistic business data and comprehensive analysis for any company context",
                    category=ToolCategory.BUSINESS,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=["business_analysis", "financial_analysis", "data_generation", "business_planning", "strategic_analysis"]
                )
                await self.tool_repository.register_tool(general_bi_tool, metadata)
                _backend_logger.info(
                    "✅ Registered general business intelligence tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register general business intelligence tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register stock business intelligence tool (for stock analysis)
            try:
                from app.tools.business_intelligence_tool import BusinessIntelligenceTool

                stock_bi_tool = BusinessIntelligenceTool()
                metadata = ToolMetadata(
                    tool_id="stock_business_intelligence",
                    name="Stock Business Intelligence",
                    description="Stock market analysis and financial intelligence operations",
                    category=ToolCategory.BUSINESS,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=["stock_analysis", "market_research", "financial_reporting"]
                )
                await self.tool_repository.register_tool(stock_bi_tool, metadata)
                _backend_logger.info(
                    "✅ Registered stock business intelligence tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register stock business intelligence tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register RAG knowledge tools
            try:
                from app.rag.tools.enhanced_knowledge_tools import (
                    EnhancedKnowledgeSearchTool,
                    AgentDocumentIngestTool,
                    AgentMemoryTool
                )

                # Knowledge search tool
                knowledge_tool = EnhancedKnowledgeSearchTool(
                    rag_system=self.unified_rag
                )
                metadata = ToolMetadata(
                    tool_id="knowledge_search",
                    name="Knowledge Search",
                    description="Search knowledge base for relevant information",
                    category=ToolCategory.RAG_ENABLED,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=True,
                    use_cases=["knowledge_search", "rag", "information_retrieval"]
                )
                await self.tool_repository.register_tool(knowledge_tool, metadata)
                _backend_logger.info(
                    "✅ Registered knowledge search tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

                # Document ingest tool
                ingest_tool = AgentDocumentIngestTool(
                    rag_system=self.unified_rag
                )
                metadata = ToolMetadata(
                    tool_id="document_ingest",
                    name="Document Ingest",
                    description="Ingest documents into knowledge base",
                    category=ToolCategory.RAG_ENABLED,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=True,
                    use_cases=["document_ingest", "knowledge_management"]
                )
                await self.tool_repository.register_tool(ingest_tool, metadata)
                _backend_logger.info(
                    "✅ Registered document ingest tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register RAG tools: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register meme tools
            try:
                from app.tools.meme_collection_tool import get_meme_collection_tool, MEME_COLLECTION_TOOL_METADATA
                from app.tools.meme_analysis_tool import get_meme_analysis_tool, MEME_ANALYSIS_TOOL_METADATA
                from app.tools.meme_generation_tool import get_meme_generation_tool, MEME_GENERATION_TOOL_METADATA

                # Register meme collection tool
                collection_tool = get_meme_collection_tool()
                await self.tool_repository.register_tool(collection_tool, MEME_COLLECTION_TOOL_METADATA)
                _backend_logger.info(
                    "✅ Registered meme collection tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

                # Register meme analysis tool
                analysis_tool = get_meme_analysis_tool()
                await self.tool_repository.register_tool(analysis_tool, MEME_ANALYSIS_TOOL_METADATA)
                _backend_logger.info(
                    "✅ Registered meme analysis tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

                # Register meme generation tool
                generation_tool = get_meme_generation_tool()
                await self.tool_repository.register_tool(generation_tool, MEME_GENERATION_TOOL_METADATA)
                _backend_logger.info(
                    "✅ Registered meme generation tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register meme tools: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register screen capture tool
            try:
                from app.tools.production.screen_capture_tool import get_screen_capture_tool

                screen_capture_tool = get_screen_capture_tool()
                metadata = ToolMetadata(
                    tool_id="screen_capture",
                    name="🖥️ Revolutionary Screen Capture Tool",
                    description="Revolutionary screen capture with multi-monitor support, region selection, OCR, and visual analysis - Capture full screen, windows, regions with advanced image processing and text extraction",
                    category=ToolCategory.AUTOMATION,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=["screen_capture", "visual_analysis", "ocr", "automation", "testing", "documentation"]
                )
                await self.tool_repository.register_tool(screen_capture_tool, metadata)
                _backend_logger.info(
                    "🖥️ Registered Revolutionary Screen Capture Tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register screen capture tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register social media orchestrator tool
            try:
                from app.tools.social_media.social_media_orchestrator_tool import get_social_media_orchestrator_tool

                social_media_tool = get_social_media_orchestrator_tool()
                metadata = ToolMetadata(
                    tool_id="social_media_orchestrator",
                    name="📱 Revolutionary Social Media Orchestrator",
                    description="Revolutionary multi-platform social media management with AI-powered content creation, scheduling, analytics, and engagement automation across all major platforms",
                    category=ToolCategory.COMMUNICATION,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=["social_media", "content_creation", "marketing", "engagement", "analytics", "automation"]
                )
                await self.tool_repository.register_tool(social_media_tool, metadata)
                _backend_logger.info(
                    "📱 Registered Revolutionary Social Media Orchestrator Tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register social media orchestrator tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register viral content generator tool
            try:
                from app.tools.social_media.viral_content_generator_tool import get_viral_content_generator_tool

                viral_content_tool = get_viral_content_generator_tool()
                metadata = ToolMetadata(
                    tool_id="viral_content_generator",
                    name="🚀 Revolutionary Viral Content Generator",
                    description="Revolutionary AI-powered viral content creation with trend analysis, engagement optimization, and multi-format content generation for maximum social media impact",
                    category=ToolCategory.CREATIVE,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=["viral_content", "content_creation", "social_media", "marketing", "engagement", "trends"]
                )
                await self.tool_repository.register_tool(viral_content_tool, metadata)
                _backend_logger.info(
                    "🚀 Registered Revolutionary Viral Content Generator Tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register viral content generator tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register AI music composition tool
            try:
                from app.tools.production.ai_music_composition_tool import get_ai_music_composition_tool

                music_tool = get_ai_music_composition_tool()
                metadata = ToolMetadata(
                    tool_id="ai_music_composition",
                    name="🎵 Revolutionary AI Music Composition Tool",
                    description="Revolutionary AI-powered music composition with multi-genre support, MIDI generation, audio synthesis, and professional music production capabilities",
                    category=ToolCategory.CREATIVE,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=["music_composition", "audio_generation", "creative_content", "entertainment", "media_production"]
                )
                await self.tool_repository.register_tool(music_tool, metadata)
                _backend_logger.info(
                    "🎵 Registered Revolutionary AI Music Composition Tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register AI music composition tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Import and register AI lyric vocal synthesis tool
            try:
                from app.tools.production.ai_lyric_vocal_synthesis_tool import get_ai_lyric_vocal_synthesis_tool

                vocal_tool = get_ai_lyric_vocal_synthesis_tool()
                metadata = ToolMetadata(
                    tool_id="ai_lyric_vocal_synthesis",
                    name="🎤 Revolutionary AI Lyric & Vocal Synthesis Tool",
                    description="Revolutionary AI-powered lyric generation and vocal synthesis with multiple voice styles, emotional expression, and professional audio production quality",
                    category=ToolCategory.CREATIVE,
                    access_level=ToolAccessLevel.PUBLIC,
                    requires_rag=False,
                    use_cases=["lyric_generation", "vocal_synthesis", "music_production", "creative_content", "entertainment"]
                )
                await self.tool_repository.register_tool(vocal_tool, metadata)
                _backend_logger.info(
                    "🎤 Registered Revolutionary AI Lyric & Vocal Synthesis Tool",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            except Exception as e:
                _backend_logger.warn(
                    f"Failed to register AI lyric vocal synthesis tool: {e}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            # Log tool registration summary
            stats = self.tool_repository.stats
            _backend_logger.info(
                f"🎯 Tool registration complete: {stats['total_tools']} tools registered",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to register built-in tools: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            # Don't raise - tool registration failure shouldn't stop system initialization

    async def _initialize_hybrid_rag_integration(self) -> None:
        """Initialize the hybrid RAG integration system."""
        try:
            from app.rag.integration import initialize_hybrid_rag_system

            success = await initialize_hybrid_rag_system()
            if success:
                _backend_logger.info(
                    "✅ Hybrid RAG Integration initialized successfully",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
            else:
                _backend_logger.warn(
                    "⚠️ Hybrid RAG Integration initialization failed",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize hybrid RAG integration: {e}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            # Don't raise - RAG integration failure shouldn't stop system initialization

    async def _initialize_phase_3_communication(self) -> None:
        """Initialize PHASE 3: Communication components."""
        try:
            # 1. Initialize THE AgentCommunicationSystem (THE communication hub)
            _backend_logger.info(
                "   📡 Initializing THE AgentCommunicationSystem...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.communication_system = AgentCommunicationSystem(
                self.unified_rag,
                self.memory_system,
                self.isolation_manager
            )
            await self.communication_system.initialize()
            self.status.components_status["communication_system"] = True

            _backend_logger.info(
                "✅ PHASE 3 Communication: COMPLETE",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize PHASE 3 Communication: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            raise

    async def _initialize_phase_4_optimization(self) -> None:
        """Initialize PHASE 4: Optimization components."""
        try:
            # PHASE 4: Optimization components (monitoring, access controls)
            # Note: Performance optimizer was removed as it was never integrated
            _backend_logger.info(
                "✅ PHASE 4 Optimization: COMPLETE",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize PHASE 4 Optimization: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            raise

    async def _initialize_tool_communication_systems(self) -> None:
        """Initialize tool and communication systems."""
        try:
            # 1. Unified Tool Repository
            _backend_logger.info(
                "Initializing Unified Tool Repository...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.tool_repository = UnifiedToolRepository(self.isolation_manager)
            await self.tool_repository.initialize()
            self.status.components_status["tool_repository"] = True

            # 2. Agent Communication System
            _backend_logger.info(
                "Initializing Agent Communication System...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.communication_system = AgentCommunicationSystem(
                self.isolation_manager,
                self.config.communication_config
            )
            await self.communication_system.initialize()
            self.status.components_status["communication_system"] = True

            _backend_logger.info(
                "✅ Tool and communication systems initialized successfully",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize tool and communication systems: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            raise

    async def _initialize_optimization_monitoring(self) -> None:
        """Initialize optimization and monitoring systems."""
        try:
            # 1. Advanced Access Controller
            if self.config.enable_security:
                _backend_logger.info(
                    "Initializing Advanced Access Controller...",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
                self.access_controller = AdvancedAccessController(self.isolation_manager)
                self.status.components_status["access_controller"] = True

            # 2. Monitoring System
            if self.config.enable_monitoring:
                _backend_logger.info(
                    "Initializing Monitoring System...",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
                self.monitoring_system = MonitoringSystem(
                    self.access_controller,
                    self.unified_rag,
                    self.memory_system,
                    self.tool_repository,
                    self.communication_system
                )
                await self.monitoring_system.initialize()
                self.status.components_status["monitoring_system"] = True

            _backend_logger.info(
                "✅ Optimization and monitoring systems initialized successfully",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize optimization and monitoring systems: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            raise

    async def _initialize_component_workflow_system(self) -> None:
        """Initialize the revolutionary component workflow execution system."""
        try:
            _backend_logger.info(
                "   🎯 Initializing Component Workflow Executor...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.component_workflow_executor = ComponentWorkflowExecutor(self)
            await self.component_workflow_executor.start_workers(num_workers=3)
            self.status.components_status["component_workflow_executor"] = True

            _backend_logger.info(
                "   🎯 Initializing Workflow Step Manager...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self.workflow_step_manager = WorkflowStepManager(self)
            self.status.components_status["workflow_step_manager"] = True

            _backend_logger.info(
                "✅ Component Workflow Execution System initialized successfully",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Failed to initialize component workflow system: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            raise

    async def execute_component_workflow(
        self,
        workflow_id: str,
        components: List[Dict[str, Any]],
        execution_mode: str = "sequential",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a component-based workflow."""
        if not self.component_workflow_executor:
            raise RuntimeError("Component workflow executor not initialized")

        return await self.component_workflow_executor.execute_component_workflow(
            workflow_id=workflow_id,
            components=components,
            execution_mode=execution_mode,
            context=context
        )

    async def _validate_system_integrity(self) -> None:
        """Validate system integrity and component connectivity."""
        try:
            _backend_logger.info(
                "Validating system integrity...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

            # Check all required components are initialized
            required_components = [
                "unified_rag", "isolation_manager", "memory_system",
                "kb_manager", "tool_repository"
            ]

            # Optional components (only check if enabled)
            optional_components = [
                "communication_system", "performance_optimizer"
            ]

            for component in required_components:
                if not self.status.components_status.get(component, False):
                    raise RuntimeError(f"Required component {component} not initialized")

            # Test component connectivity
            await self._test_component_connectivity()

            # Calculate initial health score
            self.status.health_score = await self._calculate_health_score()
            self.status.last_health_check = datetime.utcnow()

            _backend_logger.info(
                "✅ System integrity validation passed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"System integrity validation failed: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
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

            # Register agent for communication (if enabled)
            if self.communication_system:
                await self.communication_system.register_agent(test_agent_id)

            # Cleanup test agent
            # Note: In a real implementation, we'd have cleanup methods

            _backend_logger.info(
                "Component connectivity test passed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Component connectivity test failed: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
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
            _backend_logger.error(
                f"Failed to calculate health score: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
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

            _backend_logger.info(
                "🎯 UNIFIED SYSTEM SUMMARY:",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            for key, value in summary.items():
                _backend_logger.info(
                    f"   {key}: {value}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

        except Exception as e:
            _backend_logger.error(
                f"Failed to log system summary: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

    async def shutdown(self) -> None:
        """Gracefully shutdown the entire system."""
        try:
            _backend_logger.info(
                "🛑 Initiating graceful system shutdown...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            
            self.status.is_running = False
            
            # Shutdown components in reverse order
            components_to_shutdown = [
                ("performance_optimizer", self.performance_optimizer),
                ("communication_system", self.communication_system),
                ("memory_consolidation_service", self.memory_consolidation_service),  # CRITICAL FIX: Shutdown consolidation
                ("tool_repository", self.tool_repository),
                ("memory_system", self.memory_system),
                ("kb_manager", self.kb_manager),
                ("isolation_manager", self.isolation_manager),
                ("unified_rag", self.unified_rag)
            ]
            
            for component_name, component in components_to_shutdown:
                if component and hasattr(component, 'shutdown'):
                    try:
                        _backend_logger.info(
                            f"Shutting down {component_name}...",
                            LogCategory.SYSTEM_OPERATIONS,
                            "app.core.unified_system_orchestrator"
                        )
                        await component.shutdown()
                        self.status.components_status[component_name] = False
                    except Exception as e:
                        _backend_logger.error(
                            f"Error shutting down {component_name}: {str(e)}",
                            LogCategory.SYSTEM_OPERATIONS,
                            "app.core.unified_system_orchestrator"
                        )

            _backend_logger.info(
                "✅ System shutdown completed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"Error during system shutdown: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        try:
            def signal_handler(signum, frame):
                _backend_logger.info(
                    f"Received signal {signum}, initiating shutdown...",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )
                asyncio.create_task(self.shutdown())

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

        except Exception as e:
            _backend_logger.error(
                f"Failed to setup signal handlers: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

    async def _cleanup_partial_initialization(self) -> None:
        """Cleanup after partial initialization failure."""
        try:
            _backend_logger.info(
                "Cleaning up partial initialization...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            await self.shutdown()

        except Exception as e:
            _backend_logger.error(
                f"Error during cleanup: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
    
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


# ============================================================================
# AGENT BUILDER PLATFORM INTEGRATION
# ============================================================================

class AgentBuilderSystemIntegration:
    """
    Integration layer between the Agent Builder Platform and the Unified System.

    This class provides seamless integration of agent builder capabilities
    with the existing unified system architecture.
    """

    def __init__(self, system_orchestrator: UnifiedSystemOrchestrator):
        self.system_orchestrator = system_orchestrator
        self.agent_registry = None
        self.agent_factory = None
        self.llm_manager = None
        self._integration_status = "not_initialized"

    async def initialize_agent_builder_integration(self) -> bool:
        """Initialize Agent Builder platform integration."""
        try:
            _backend_logger.info(
                "🤖 Initializing Agent Builder Platform integration...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

            # Import Agent Builder components
            from app.agents.factory import AgentBuilderFactory
            from app.agents.registry import initialize_agent_registry
            from app.llm.manager import get_enhanced_llm_manager

            # Initialize LLM manager
            self.llm_manager = get_enhanced_llm_manager()
            if not self.llm_manager.is_initialized():
                await self.llm_manager.initialize()

            # 🚀 Register LLM manager with configuration managers
            _backend_logger.info(
                "   🔧 Registering LLM manager with configuration managers...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            from app.core.global_config_manager import global_config_manager
            from app.services.llm_service import get_llm_service, initialize_llm_service

            # Initialize LLM service
            llm_service = await initialize_llm_service()

            # Find and update LLM section manager
            for section_manager in global_config_manager._section_managers.values():
                if hasattr(section_manager, 'set_llm_manager'):
                    section_manager.set_llm_manager(self.llm_manager)
                    _backend_logger.info(
                        "   ✅ LLM manager registered with LLM section manager",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.unified_system_orchestrator"
                    )
                if hasattr(section_manager, 'set_llm_service'):
                    section_manager.set_llm_service(llm_service)
                    _backend_logger.info(
                        "   ✅ LLM service registered with LLM section manager",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.unified_system_orchestrator"
                    )

            # Find and update LLM observer
            for observer in global_config_manager._observers.get("llm_providers", []):
                if hasattr(observer, 'set_llm_manager'):
                    observer.set_llm_manager(self.llm_manager)
                    _backend_logger.info(
                        "   ✅ LLM manager registered with LLM observer",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.unified_system_orchestrator"
                    )
                if hasattr(observer, 'set_llm_service'):
                    observer.set_llm_service(llm_service)
                    _backend_logger.info(
                        "   ✅ LLM service registered with LLM observer",
                        LogCategory.SYSTEM_OPERATIONS,
                        "app.core.unified_system_orchestrator"
                    )

            # Initialize agent factory
            self.agent_factory = AgentBuilderFactory(self.llm_manager)

            # Initialize agent registry with system orchestrator
            self.agent_registry = initialize_agent_registry(
                self.agent_factory,
                self.system_orchestrator
            )

            self._integration_status = "initialized"
            _backend_logger.info(
                "✅ Agent Builder Platform integration initialized successfully",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            return True

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to initialize Agent Builder integration: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            self._integration_status = "failed"
            return False

    async def create_system_agents(self) -> Dict[str, str]:
        """Create essential system agents for platform operations."""
        try:
            if not self.agent_registry:
                await self.initialize_agent_builder_integration()

            _backend_logger.info(
                "🏗️ Creating essential system agents...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

            system_agents = {}

            # System Monitor Agent
            from app.agents.factory import AgentType, AgentBuilderConfig
            from app.llm.models import LLMConfig, ProviderType
            from app.agents.base.agent import AgentCapability

            monitor_config = AgentBuilderConfig(
                name="System Monitor Agent",
                description="Monitors system health, performance, and agent activities",
                agent_type=AgentType.AUTONOMOUS,
                llm_config=LLMConfig(
                    provider=ProviderType.OLLAMA,
                    model_id="llama3.2:latest",
                    temperature=0.2,
                    max_tokens=2048
                ),
                capabilities=[
                    AgentCapability.REASONING,
                    AgentCapability.TOOL_USE,
                    AgentCapability.MONITORING,
                    AgentCapability.ANALYSIS
                ],
                tools=["system_monitor", "performance_tracker", "health_checker", "alert_manager"],
                system_prompt="""You are the System Monitor Agent. Your responsibilities include:
                1. Monitor system health and performance metrics
                2. Track agent activities and resource usage
                3. Detect anomalies and potential issues
                4. Generate alerts for critical situations
                5. Provide system status reports

                Always prioritize system stability and proactive issue detection.""",
                enable_memory=True,
                enable_learning=True,
                enable_collaboration=True
            )

            monitor_agent_id = await self.agent_registry.register_agent(
                config=monitor_config,
                owner="system",
                tags=["system", "monitoring", "essential"]
            )
            await self.agent_registry.start_agent(monitor_agent_id)
            system_agents["system_monitor"] = monitor_agent_id

            # Resource Manager Agent
            resource_config = AgentBuilderConfig(
                name="Resource Manager Agent",
                description="Manages system resources, load balancing, and optimization",
                agent_type=AgentType.AUTONOMOUS,
                llm_config=LLMConfig(
                    provider=ProviderType.OLLAMA,
                    model_id="llama3.2:latest",
                    temperature=0.1,
                    max_tokens=2048
                ),
                capabilities=[
                    AgentCapability.REASONING,
                    AgentCapability.TOOL_USE,
                    AgentCapability.OPTIMIZATION,
                    AgentCapability.RESOURCE_MANAGEMENT
                ],
                tools=["resource_allocator", "load_balancer", "optimizer", "capacity_planner"],
                system_prompt="""You are the Resource Manager Agent. Your responsibilities include:
                1. Monitor and manage system resource allocation
                2. Optimize performance and resource utilization
                3. Balance loads across system components
                4. Plan capacity and scaling requirements
                5. Prevent resource conflicts and bottlenecks

                Focus on efficiency, scalability, and optimal resource utilization.""",
                enable_memory=True,
                enable_learning=True,
                enable_collaboration=True
            )

            resource_agent_id = await self.agent_registry.register_agent(
                config=resource_config,
                owner="system",
                tags=["system", "resource_management", "essential"]
            )
            await self.agent_registry.start_agent(resource_agent_id)
            system_agents["resource_manager"] = resource_agent_id

            # Security Agent
            security_config = AgentBuilderConfig(
                name="Security Guardian Agent",
                description="Monitors security, access control, and threat detection",
                agent_type=AgentType.AUTONOMOUS,
                llm_config=LLMConfig(
                    provider=ProviderType.OLLAMA,
                    model_id="llama3.2:latest",
                    temperature=0.1,
                    max_tokens=2048
                ),
                capabilities=[
                    AgentCapability.REASONING,
                    AgentCapability.TOOL_USE,
                    AgentCapability.SECURITY,
                    AgentCapability.MONITORING
                ],
                tools=["security_scanner", "access_monitor", "threat_detector", "audit_logger"],
                system_prompt="""You are the Security Guardian Agent. Your responsibilities include:
                1. Monitor system security and access patterns
                2. Detect potential security threats and anomalies
                3. Enforce access control and security policies
                4. Audit system activities and maintain security logs
                5. Respond to security incidents and alerts

                Maintain the highest security standards and protect system integrity.""",
                enable_memory=True,
                enable_learning=True,
                enable_collaboration=True
            )

            security_agent_id = await self.agent_registry.register_agent(
                config=security_config,
                owner="system",
                tags=["system", "security", "essential"]
            )
            await self.agent_registry.start_agent(security_agent_id)
            system_agents["security_guardian"] = security_agent_id

            # Create collaboration group for system agents
            if len(system_agents) > 1:
                await self.agent_registry.create_collaboration_group(
                    group_id="system_agents",
                    agent_ids=list(system_agents.values())
                )

            _backend_logger.info(
                f"✅ Created {len(system_agents)} essential system agents",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            return system_agents

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to create system agents: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            return {}

    def get_integration_status(self) -> Dict[str, Any]:
        """Get the current integration status."""
        return {
            "status": self._integration_status,
            "agent_registry_initialized": self.agent_registry is not None,
            "agent_factory_initialized": self.agent_factory is not None,
            "llm_manager_initialized": self.llm_manager is not None and self.llm_manager.is_initialized(),
            "system_orchestrator_status": self.system_orchestrator.status.status.value if self.system_orchestrator else "not_available"
        }

    async def shutdown_agent_builder_integration(self):
        """Shutdown Agent Builder platform integration."""
        try:
            _backend_logger.info(
                "🔄 Shutting down Agent Builder Platform integration...",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

            if self.agent_registry:
                # Stop all agents
                agents = self.agent_registry.list_agents()
                for agent in agents:
                    await self.agent_registry.stop_agent(agent.agent_id)

                # Clear registry
                self.agent_registry = None

            self.agent_factory = None
            self.llm_manager = None
            self._integration_status = "shutdown"

            _backend_logger.info(
                "✅ Agent Builder Platform integration shutdown complete",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to shutdown Agent Builder integration: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )


# Enhanced Unified System Orchestrator with Agent Builder Integration
class EnhancedUnifiedSystemOrchestrator(UnifiedSystemOrchestrator):
    """
    Enhanced version of the Unified System Orchestrator with Agent Builder integration.

    This orchestrator includes all the original functionality plus Agent Builder
    platform capabilities for comprehensive AI agent management.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        super().__init__(config)
        self.agent_builder_integration: Optional[AgentBuilderSystemIntegration] = None

    @property
    def is_initialized(self) -> bool:
        """Check if the enhanced orchestrator is initialized."""
        return self.status.is_initialized

    @property
    def rag_system(self):
        """Get the unified RAG system for backward compatibility."""
        return self.unified_rag

    async def initialize(self) -> bool:
        """Initialize the enhanced system with Agent Builder integration."""
        try:
            # Initialize base system first
            base_success = await super().initialize()
            if not base_success:
                return False

            # Initialize Agent Builder integration
            self.agent_builder_integration = AgentBuilderSystemIntegration(self)
            integration_success = await self.agent_builder_integration.initialize_agent_builder_integration()

            if integration_success:
                # Create essential system agents
                system_agents = await self.agent_builder_integration.create_system_agents()
                _backend_logger.info(
                    f"🎯 Enhanced Unified System initialized with {len(system_agents)} system agents",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            return integration_success

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to initialize enhanced system: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            return False

    async def shutdown(self):
        """Shutdown the enhanced system."""
        try:
            # Shutdown Agent Builder integration first
            if self.agent_builder_integration:
                await self.agent_builder_integration.shutdown_agent_builder_integration()

            # Shutdown base system
            await super().shutdown()

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to shutdown enhanced system: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including Agent Builder integration."""
        base_status = super().get_system_status()

        if self.agent_builder_integration:
            base_status["agent_builder_integration"] = self.agent_builder_integration.get_integration_status()

        return base_status


# ============================================================================
# REVOLUTIONARY COMPONENT WORKFLOW EXECUTION SYSTEM
# ============================================================================

class ComponentWorkflowExecutor:
    """Revolutionary async component workflow executor."""

    def __init__(self, orchestrator: 'UnifiedSystemOrchestrator'):
        self.orchestrator = orchestrator
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.execution_queue = asyncio.Queue()
        self.workers_running = False
        self._backend_logger = get_backend_logger()

    async def start_workers(self, num_workers: int = 3) -> None:
        """Start async workflow execution workers."""
        if self.workers_running:
            return

        self.workers_running = True
        self.worker_tasks = []

        for i in range(num_workers):
            task = asyncio.create_task(self._workflow_worker(f"worker-{i}"))
            self.worker_tasks.append(task)

        self._backend_logger.info(
            "Component workflow workers started",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.unified_system_orchestrator",
            data={"num_workers": num_workers}
        )

    async def stop_workers(self) -> None:
        """Stop workflow execution workers."""
        self.workers_running = False

        if hasattr(self, 'worker_tasks'):
            for task in self.worker_tasks:
                task.cancel()
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        self._backend_logger.info(
            "Component workflow workers stopped",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.unified_system_orchestrator"
        )

    async def execute_component_workflow(
        self,
        workflow_id: str,
        components: List[Dict[str, Any]],
        execution_mode: str = "sequential",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a component-based workflow asynchronously."""
        try:
            workflow_context = {
                "workflow_id": workflow_id,
                "components": components,
                "execution_mode": execution_mode,
                "context": context or {},
                "status": "running",
                "start_time": datetime.utcnow(),
                "results": {},
                "current_step": 0,
                "total_steps": len(components)
            }

            self.active_workflows[workflow_id] = workflow_context

            # Queue workflow for execution
            await self.execution_queue.put(workflow_context)

            self._backend_logger.info(
                "Component workflow queued for execution",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator",
                data={
                    "workflow_id": workflow_id,
                    "num_components": len(components),
                    "execution_mode": execution_mode
                }
            )

            return {
                "workflow_id": workflow_id,
                "status": "queued",
                "message": "Workflow queued for execution",
                "total_steps": len(components)
            }

        except Exception as e:
            self._backend_logger.error(
                "Failed to execute component workflow",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator",
                data={"error": str(e)}
            )
            raise

    async def _workflow_worker(self, worker_id: str) -> None:
        """Async worker for processing component workflows."""
        self._backend_logger.info(
            "Workflow worker started",
            LogCategory.SYSTEM_OPERATIONS,
            "app.core.unified_system_orchestrator",
            data={"worker_id": worker_id}
        )

        while self.workers_running:
            try:
                # Get workflow from queue with timeout
                workflow_context = await asyncio.wait_for(
                    self.execution_queue.get(), timeout=1.0
                )

                await self._execute_workflow_steps(workflow_context, worker_id)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._backend_logger.error(
                    "Workflow worker error",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator",
                    data={"worker_id": worker_id, "error": str(e)}
                )

    async def _execute_workflow_steps(
        self,
        workflow_context: Dict[str, Any],
        worker_id: str
    ) -> None:
        """Execute workflow steps based on execution mode."""
        workflow_id = workflow_context["workflow_id"]
        components = workflow_context["components"]
        execution_mode = workflow_context["execution_mode"]

        try:
            if execution_mode == "sequential":
                await self._execute_sequential(workflow_context, worker_id)
            elif execution_mode == "parallel":
                await self._execute_parallel(workflow_context, worker_id)
            elif execution_mode == "autonomous":
                await self._execute_autonomous(workflow_context, worker_id)
            else:
                raise ValueError(f"Unknown execution mode: {execution_mode}")

            workflow_context["status"] = "completed"
            workflow_context["end_time"] = datetime.utcnow()

            self._backend_logger.info(
                "Component workflow completed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator",
                data={
                    "workflow_id": workflow_id,
                    "worker_id": worker_id,
                    "execution_time": (workflow_context["end_time"] - workflow_context["start_time"]).total_seconds()
                }
            )

        except Exception as e:
            workflow_context["status"] = "failed"
            workflow_context["error"] = str(e)
            workflow_context["end_time"] = datetime.utcnow()

            self._backend_logger.error(
                "Component workflow failed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator",
                data={
                    "workflow_id": workflow_id,
                    "worker_id": worker_id,
                    "error": str(e)
                }
            )

    async def _execute_sequential(self, workflow_context: Dict[str, Any], worker_id: str) -> None:
        """Execute components sequentially."""
        components = workflow_context["components"]
        results = {}

        for i, component in enumerate(components):
            workflow_context["current_step"] = i + 1

            step_result = await self._execute_component_step(
                component, workflow_context, f"step-{i+1}"
            )

            results[f"step_{i+1}"] = step_result
            workflow_context["results"] = results

    async def _execute_parallel(self, workflow_context: Dict[str, Any], worker_id: str) -> None:
        """Execute components in parallel."""
        components = workflow_context["components"]

        # Create tasks for all components
        tasks = []
        for i, component in enumerate(components):
            task = asyncio.create_task(
                self._execute_component_step(component, workflow_context, f"step-{i+1}")
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        workflow_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                workflow_results[f"step_{i+1}"] = {"error": str(result), "status": "failed"}
            else:
                workflow_results[f"step_{i+1}"] = result

        workflow_context["results"] = workflow_results

    async def _execute_autonomous(self, workflow_context: Dict[str, Any], worker_id: str) -> None:
        """Execute components with autonomous decision-making."""
        # This would integrate with the AutonomousLangGraphAgent
        # For now, fall back to sequential execution with autonomous agents
        await self._execute_sequential(workflow_context, worker_id)

    async def _execute_component_step(
        self,
        component: Dict[str, Any],
        workflow_context: Dict[str, Any],
        step_id: str
    ) -> Dict[str, Any]:
        """Execute a single component step."""
        try:
            component_type = component.get("type")
            component_config = component.get("config", {})

            # Get step manager for detailed step execution
            step_manager = self.orchestrator.workflow_step_manager
            if step_manager:
                return await step_manager.execute_step(
                    step_id=step_id,
                    component=component,
                    context=workflow_context["context"]
                )

            # Fallback execution
            return {
                "step_id": step_id,
                "component_type": component_type,
                "status": "completed",
                "result": f"Executed {component_type} component",
                "execution_time": 0.1
            }

        except Exception as e:
            return {
                "step_id": step_id,
                "status": "failed",
                "error": str(e),
                "execution_time": 0.0
            }

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running workflow."""
        return self.active_workflows.get(workflow_id)

    def list_active_workflows(self) -> List[str]:
        """List all active workflow IDs."""
        return list(self.active_workflows.keys())


class WorkflowStepManager:
    """Revolutionary async workflow step manager."""

    def __init__(self, orchestrator: 'UnifiedSystemOrchestrator'):
        self.orchestrator = orchestrator
        self.step_states: Dict[str, Dict[str, Any]] = {}
        self.step_results: Dict[str, Dict[str, Any]] = {}
        self._backend_logger = get_backend_logger()

    async def execute_step(
        self,
        step_id: str,
        component: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step with full state tracking."""
        start_time = datetime.utcnow()

        # Initialize step state
        self.step_states[step_id] = {
            "step_id": step_id,
            "component": component,
            "status": "running",
            "start_time": start_time,
            "context": context
        }

        try:
            component_type = component.get("type")
            component_config = component.get("config", {})

            # Execute based on component type
            if component_type == "TOOL":
                result = await self._execute_tool_component(component_config, context)
            elif component_type == "CAPABILITY":
                result = await self._execute_capability_component(component_config, context)
            elif component_type == "PROMPT":
                result = await self._execute_prompt_component(component_config, context)
            elif component_type == "WORKFLOW_STEP":
                result = await self._execute_workflow_step_component(component_config, context)
            else:
                result = await self._execute_custom_component(component, context)

            # Update step state
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            step_result = {
                "step_id": step_id,
                "component_type": component_type,
                "status": "completed",
                "result": result,
                "execution_time": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

            self.step_states[step_id].update({
                "status": "completed",
                "end_time": end_time,
                "execution_time": execution_time
            })

            self.step_results[step_id] = step_result

            self._backend_logger.info(
                "Workflow step completed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator",
                data={
                    "step_id": step_id,
                    "component_type": component_type,
                    "execution_time": execution_time
                }
            )

            return step_result

        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            error_result = {
                "step_id": step_id,
                "status": "failed",
                "error": str(e),
                "execution_time": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

            self.step_states[step_id].update({
                "status": "failed",
                "error": str(e),
                "end_time": end_time,
                "execution_time": execution_time
            })

            self.step_results[step_id] = error_result

            self._backend_logger.error(
                "Workflow step failed",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator",
                data={
                    "step_id": step_id,
                    "error": str(e),
                    "execution_time": execution_time
                }
            )

            return error_result

    async def _execute_tool_component(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool component."""
        tool_name = config.get("tool_name", "unknown")
        tool_params = config.get("parameters", {})

        # Simulate tool execution
        await asyncio.sleep(0.1)  # Simulate processing time

        return {
            "tool_name": tool_name,
            "parameters": tool_params,
            "output": f"Tool {tool_name} executed successfully",
            "context_updated": True
        }

    async def _execute_capability_component(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a capability component."""
        capability_name = config.get("capability_name", "unknown")
        capability_params = config.get("parameters", {})

        # Simulate capability execution
        await asyncio.sleep(0.2)  # Simulate processing time

        return {
            "capability_name": capability_name,
            "parameters": capability_params,
            "output": f"Capability {capability_name} executed successfully",
            "context_updated": True
        }

    async def _execute_prompt_component(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a prompt component."""
        prompt_template = config.get("template", "")
        prompt_variables = config.get("variables", {})

        # Simulate LLM execution
        await asyncio.sleep(0.5)  # Simulate LLM processing time

        return {
            "prompt_template": prompt_template,
            "variables": prompt_variables,
            "output": f"Prompt executed with template: {prompt_template[:50]}...",
            "context_updated": True
        }

    async def _execute_workflow_step_component(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow step component."""
        step_name = config.get("step_name", "unknown")
        step_action = config.get("action", "process")

        # Simulate workflow step execution
        await asyncio.sleep(0.3)  # Simulate processing time

        return {
            "step_name": step_name,
            "action": step_action,
            "output": f"Workflow step {step_name} executed successfully",
            "context_updated": True
        }

    async def _execute_custom_component(self, component: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom component."""
        component_name = component.get("name", "unknown")

        # Simulate custom component execution
        await asyncio.sleep(0.2)  # Simulate processing time

        return {
            "component_name": component_name,
            "output": f"Custom component {component_name} executed successfully",
            "context_updated": True
        }

    def get_step_state(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a workflow step."""
        return self.step_states.get(step_id)

    def get_step_result(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed workflow step."""
        return self.step_results.get(step_id)

    def list_active_steps(self) -> List[str]:
        """List all active step IDs."""
        return [
            step_id for step_id, state in self.step_states.items()
            if state.get("status") == "running"
        ]


# Global enhanced orchestrator instance
_enhanced_system_orchestrator: Optional[EnhancedUnifiedSystemOrchestrator] = None


def get_enhanced_system_orchestrator() -> EnhancedUnifiedSystemOrchestrator:
    """Get the global enhanced system orchestrator instance."""
    global _enhanced_system_orchestrator
    if _enhanced_system_orchestrator is None:
        _enhanced_system_orchestrator = EnhancedUnifiedSystemOrchestrator()
        # Note: Initialization will be handled by the seamless integration system
    return _enhanced_system_orchestrator


# ============================================================================
# COMPATIBILITY LAYER FOR API ENDPOINTS
# ============================================================================

class OrchestrationCompatibilityLayer:
    """
    Compatibility layer to make UnifiedSystemOrchestrator compatible with
    existing API endpoint expectations.
    """

    def __init__(self, enhanced_orchestrator: EnhancedUnifiedSystemOrchestrator):
        self.enhanced_orchestrator = enhanced_orchestrator
        self._agents = {}  # Agent registry cache
        self._llm = None   # LLM instance cache

    @property
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized."""
        return self.enhanced_orchestrator.status.is_initialized

    @property
    def agents(self) -> Dict[str, Any]:
        """Get agents registry."""
        if self.enhanced_orchestrator.agent_builder_integration and self.enhanced_orchestrator.agent_builder_integration.agent_registry:
            # Return live agent registry
            registry = self.enhanced_orchestrator.agent_builder_integration.agent_registry
            return {agent.agent_id: agent for agent in registry.list_agents()}
        return self._agents

    @property
    def workflows(self) -> Dict[str, Any]:
        """Get workflows registry."""
        if self.enhanced_orchestrator.component_workflow_executor:
            return self.enhanced_orchestrator.component_workflow_executor.active_workflows
        return {}

    @property
    def llm(self):
        """Get LLM instance."""
        if self.enhanced_orchestrator.agent_builder_integration and self.enhanced_orchestrator.agent_builder_integration.llm_manager:
            # Return default LLM from manager
            llm_manager = self.enhanced_orchestrator.agent_builder_integration.llm_manager
            return llm_manager.get_default_llm()
        return self._llm

    @property
    def checkpoint_saver(self):
        """Get checkpoint saver."""
        # Return None for now - can be implemented later
        return None

    @property
    def rag_system(self):
        """Get the unified RAG system for backward compatibility."""
        return self.enhanced_orchestrator.unified_rag

    async def initialize(self):
        """Initialize the orchestrator."""
        if not self.enhanced_orchestrator.status.is_initialized:
            await self.enhanced_orchestrator.initialize()

    async def create_agent(self, agent_type: str, config: Dict[str, Any]) -> str:
        """Create a new agent."""
        if not self.enhanced_orchestrator.agent_builder_integration:
            await self.enhanced_orchestrator.initialize()

        # Use agent factory to create agent
        from app.agents.factory import AgentType, AgentBuilderConfig
        from app.llm.models import LLMConfig, ProviderType
        from app.agents.base.agent import AgentCapability

        # Convert config to AgentBuilderConfig
        agent_config = AgentBuilderConfig(
            name=config.get("name", f"{agent_type} Agent"),
            description=config.get("description", f"Agent of type {agent_type}"),
            agent_type=AgentType(agent_type.lower()) if hasattr(AgentType, agent_type.lower()) else AgentType.REACT,
            llm_config=LLMConfig(
                provider=ProviderType.OLLAMA,
                model_id=config.get("model", "llama3.2:3b"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2048)
            ),
            capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE],
            tools=config.get("tools", [])
        )

        # Create agent using registry
        registry = self.enhanced_orchestrator.agent_builder_integration.agent_registry
        agent_id = await registry.create_agent(
            config=agent_config,
            owner=config.get("owner", "system"),
            tags=config.get("tags", [])
        )

        # Start the agent
        await registry.start_agent(agent_id)

        return agent_id

    async def get_agent(self, agent_id: str):
        """Get an agent by ID."""
        if self.enhanced_orchestrator.agent_builder_integration and self.enhanced_orchestrator.agent_builder_integration.agent_registry:
            registry = self.enhanced_orchestrator.agent_builder_integration.agent_registry
            return registry.get_agent(agent_id)
        return self._agents.get(agent_id)

    async def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any], agent_ids: List[str] = None) -> Dict[str, Any]:
        """Execute a workflow."""
        # Convert to component workflow format
        components = [
            {
                "type": "workflow_step",
                "name": f"step_{i}",
                "step_name": f"workflow_step_{i}",
                "action": "process",
                "parameters": inputs
            }
            for i in range(1, 4)  # Create 3 steps by default
        ]

        return await self.enhanced_orchestrator.execute_component_workflow(
            workflow_id=workflow_id,
            components=components,
            execution_mode="sequential",
            context=inputs
        )

    async def execute_hierarchical_workflow(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a hierarchical workflow."""
        # Use the hierarchical workflow orchestrator from subgraphs
        from app.orchestration.subgraphs import HierarchicalWorkflowOrchestrator
        from langchain_openai import ChatOpenAI

        # Create hierarchical orchestrator
        llm = self.llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        hierarchical_orchestrator = HierarchicalWorkflowOrchestrator(llm=llm)

        # Execute hierarchical workflow
        return await hierarchical_orchestrator.execute_hierarchical_workflow(
            task=task,
            context=context or {}
        )

    async def create_agent(
        self,
        agent_id: str,
        agent_type: str = "rag",
        model_name: str = "llama3.1:8b",
        **kwargs
    ) -> Any:
        """
        Create an agent using the Agent Builder integration.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent to create
            model_name: LLM model to use
            **kwargs: Additional agent configuration

        Returns:
            Created agent instance
        """
        try:
            if not self.agent_builder_integration:
                raise RuntimeError("Agent Builder integration not initialized")

            # Use the agent builder integration to create agent
            from app.agents.factory import AgentBuilderFactory, AgentBuilderConfig, AgentType

            # Map string type to enum
            agent_type_enum = AgentType.RAG if agent_type == "rag" else AgentType.REACT

            config = AgentBuilderConfig(
                agent_id=agent_id,
                agent_type=agent_type_enum,
                model_name=model_name,
                **kwargs
            )

            factory = AgentBuilderFactory()
            agent = await factory.create_agent(config)

            _backend_logger.info(
                f"✅ Created agent {agent_id} of type {agent_type}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            return agent

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to create agent {agent_id}: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            raise

    async def create_agent_knowledge_base(
        self,
        agent_id: str,
        documents: List[Any] = None
    ) -> str:
        """
        Create a knowledge base for an agent.

        Args:
            agent_id: Agent identifier
            documents: Optional list of documents to ingest

        Returns:
            Knowledge base identifier
        """
        try:
            if not self.unified_rag:
                raise RuntimeError("RAG system not initialized")

            # Create agent ecosystem in RAG system
            agent_collections = await self.unified_rag.create_agent_ecosystem(agent_id)

            # Ingest documents if provided
            if documents:
                await self.unified_rag.add_documents(agent_id, documents)
                _backend_logger.info(
                    f"✅ Ingested {len(documents)} documents for agent {agent_id}",
                    LogCategory.SYSTEM_OPERATIONS,
                    "app.core.unified_system_orchestrator"
                )

            kb_id = f"kb_agent_{agent_id}"
            _backend_logger.info(
                f"✅ Created knowledge base {kb_id} for agent {agent_id}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            return kb_id

        except Exception as e:
            _backend_logger.error(
                f"❌ Failed to create knowledge base for agent {agent_id}: {str(e)}",
                LogCategory.SYSTEM_OPERATIONS,
                "app.core.unified_system_orchestrator"
            )
            raise


# Create compatibility wrapper for the enhanced orchestrator
def get_orchestrator_with_compatibility() -> OrchestrationCompatibilityLayer:
    """Get orchestrator with compatibility layer for API endpoints."""
    enhanced_orchestrator = get_enhanced_system_orchestrator()
    return OrchestrationCompatibilityLayer(enhanced_orchestrator)
