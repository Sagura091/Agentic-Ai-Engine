"""
Unified Tool Repository - THE Tool System for Multi-Agent Architecture.

This is THE ONLY tool repository in the entire application.
All tool operations flow through this unified repository.

CORE ARCHITECTURE:
- Centralized tool registry with agent-specific access
- Dynamic tool assignment based on use cases
- RAG-enabled and non-RAG tools
- Simple, clean, fast operations

DESIGN PRINCIPLES:
- One tool repository to rule them all
- Use case driven tool access
- Agent-specific tool permissions
- No complexity unless absolutely necessary

PHASE 2 ENHANCEMENT:
✅ Integration with UnifiedRAGSystem
✅ Dynamic tool assignment
✅ Agent-specific tool access
✅ Use case based tool selection
"""

import asyncio
from typing import Dict, List, Optional, Any, Set, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

logger = structlog.get_logger(__name__)


class ToolCategory(str, Enum):
    """Categories of tools in the repository - ENHANCED WITH AUTOMATION AND TRADING."""
    RAG_ENABLED = "rag_enabled"       # Tools that use RAG system
    COMPUTATION = "computation"       # Calculator, math tools
    COMMUNICATION = "communication"   # Agent communication tools
    RESEARCH = "research"             # Web search, research tools
    BUSINESS = "business"             # Business analysis tools
    UTILITY = "utility"               # File operations, utilities
    DATA = "data"                     # Database operations, data management
    ANALYSIS = "analysis"             # Text processing, NLP, analytics
    SECURITY = "security"             # Password, security, authentication
    AUTOMATION = "automation"         # Browser automation, desktop automation, visual analysis
    TRADING = "trading"               # Stock trading, financial analysis, market data
    PRODUCTIVITY = "productivity"     # Document generation, file creation, productivity tools
    CREATIVE = "creative"             # Music, art, content creation, creative tools
    SOCIAL_MEDIA = "social_media"     # Social media management, content distribution


class ToolAccessLevel(str, Enum):
    """Access levels for tools - SIMPLIFIED."""
    PUBLIC = "public"                 # Available to all agents
    PRIVATE = "private"               # Agent-specific tools
    CONDITIONAL = "conditional"       # Based on agent configuration


@dataclass
class ToolMetadata:
    """Simple metadata for a tool in the repository - ENHANCED."""
    tool_id: str
    name: str
    description: str
    category: ToolCategory
    access_level: ToolAccessLevel
    requires_rag: bool = False           # NEW: Does this tool need RAG access?
    use_cases: Set[str] = field(default_factory=set)  # NEW: Use cases for this tool
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    is_active: bool = True


@dataclass
class AgentToolProfile:
    """Simple tool profile for an agent - ENHANCED."""
    agent_id: str
    assigned_tools: Set[str] = field(default_factory=set)
    usage_stats: Dict[str, int] = field(default_factory=dict)
    rag_enabled: bool = True             # NEW: Does this agent have RAG access?
    allowed_categories: Set[ToolCategory] = field(default_factory=set)  # NEW: Allowed tool categories
    last_updated: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(cls, agent_id: str, rag_enabled: bool = True) -> "AgentToolProfile":
        """Create a new agent tool profile."""
        return cls(
            agent_id=agent_id,
            rag_enabled=rag_enabled,
            allowed_categories=set(ToolCategory)  # Allow all categories by default
        )


class UnifiedToolRepository:
    """
    Unified Tool Repository - THE Tool System.

    SIMPLIFIED ARCHITECTURE:
    - Dynamic tool assignment based on use cases
    - RAG-enabled vs non-RAG tools
    - Agent-specific tool access
    - Use case driven tool selection
    """

    def __init__(self, unified_rag=None, isolation_manager=None):
        """Initialize THE unified tool repository."""
        self.unified_rag = unified_rag
        self.isolation_manager = isolation_manager

        # Tool registry - ENHANCED
        self.tools: Dict[str, BaseTool] = {}                    # tool_id -> tool_instance
        self.tool_metadata: Dict[str, ToolMetadata] = {}        # tool_id -> metadata

        # Agent profiles - ENHANCED
        self.agent_profiles: Dict[str, AgentToolProfile] = {}   # agent_id -> profile

        # Use case mapping - NEW
        self.use_case_tools: Dict[str, Set[str]] = {}           # use_case -> tool_ids

        # Simple stats
        self.stats = {
            "total_tools": 0,
            "total_agents": 0,
            "total_tool_calls": 0,
            "rag_enabled_tools": 0,
            "tools_by_category": {}
        }

        self.is_initialized = False
        logger.info("THE Unified tool repository created")
    
    async def initialize(self) -> None:
        """Initialize the tool repository."""
        try:
            if self.is_initialized:
                return

            # Initialize category stats
            for category in ToolCategory:
                self.stats["tools_by_category"][category.value] = 0

            self.is_initialized = True
            logger.info("Tool repository initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize tool repository: {str(e)}")
            raise

    async def register_tool(
        self,
        tool_instance: BaseTool,
        metadata: ToolMetadata
    ) -> str:
        """
        Register a new tool in the repository - ENHANCED.

        Args:
            tool_instance: Tool instance to register
            metadata: Tool metadata

        Returns:
            Tool ID
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            tool_id = metadata.tool_id

            if tool_id in self.tools:
                logger.warning(f"Tool {tool_id} already registered")
                return tool_id

            # Register tool
            self.tools[tool_id] = tool_instance
            self.tool_metadata[tool_id] = metadata

            # Update use case mapping - NEW
            for use_case in metadata.use_cases:
                if use_case not in self.use_case_tools:
                    self.use_case_tools[use_case] = set()
                self.use_case_tools[use_case].add(tool_id)

            # Update stats
            self.stats["total_tools"] += 1
            self.stats["tools_by_category"][metadata.category.value] += 1
            if metadata.requires_rag:
                self.stats["rag_enabled_tools"] += 1

            logger.debug(f"Registered tool: {tool_id} ({metadata.category.value}, RAG: {metadata.requires_rag})")
            return tool_id

        except Exception as e:
            logger.error(f"Failed to register tool {metadata.tool_id}: {str(e)}")
            raise
    
    async def get_tools_for_use_case(
        self,
        agent_id: str,
        use_cases: List[str],
        include_rag_tools: bool = True
    ) -> List[BaseTool]:
        """
        Get tools for an agent based on use cases - DYNAMIC TOOL SELECTION.

        Args:
            agent_id: Agent requesting tools
            use_cases: List of use cases (e.g., ["knowledge_search", "calculation"])
            include_rag_tools: Whether to include RAG-enabled tools

        Returns:
            List of tools for the agent
        """
        try:
            if not self.is_initialized:
                await self.initialize()

            # Get or create agent profile
            if agent_id not in self.agent_profiles:
                await self.create_agent_profile(agent_id)

            profile = self.agent_profiles[agent_id]
            selected_tools = []

            # Get tools for each use case
            for use_case in use_cases:
                if use_case in self.use_case_tools:
                    for tool_id in self.use_case_tools[use_case]:
                        metadata = self.tool_metadata[tool_id]

                        # Check if agent can access this tool
                        if not self._can_agent_access_tool(agent_id, tool_id):
                            continue

                        # Check RAG requirements
                        if metadata.requires_rag and not include_rag_tools:
                            continue

                        if metadata.requires_rag and not profile.rag_enabled:
                            continue

                        # Add tool if not already added
                        tool = self.tools[tool_id]
                        if tool not in selected_tools:
                            selected_tools.append(tool)
                else:
                    # Handle new use cases by loading appropriate tools
                    await self._load_tools_for_new_use_case(use_case)

                    # Retry after loading
                    if use_case in self.use_case_tools:
                        for tool_id in self.use_case_tools[use_case]:
                            metadata = self.tool_metadata[tool_id]

                            # Check if agent can access this tool
                            if not self._can_agent_access_tool(agent_id, tool_id):
                                continue

                            # Check RAG requirements
                            if metadata.requires_rag and not include_rag_tools:
                                continue

                            if metadata.requires_rag and not profile.rag_enabled:
                                continue

                            # Add tool if not already added
                            tool = self.tools[tool_id]
                            if tool not in selected_tools:
                                selected_tools.append(tool)

            logger.debug(f"Selected {len(selected_tools)} tools for agent {agent_id} with use cases: {use_cases}")
            return selected_tools

        except Exception as e:
            logger.error(f"Failed to get tools for agent {agent_id}: {str(e)}")
            return []

    def _can_agent_access_tool(self, agent_id: str, tool_id: str) -> bool:
        """Check if an agent can access a specific tool."""
        metadata = self.tool_metadata[tool_id]

        # Public tools are always accessible
        if metadata.access_level == ToolAccessLevel.PUBLIC:
            return True

        # Private tools need explicit assignment
        if metadata.access_level == ToolAccessLevel.PRIVATE:
            profile = self.agent_profiles.get(agent_id)
            return profile and tool_id in profile.assigned_tools

        # Conditional tools based on agent configuration
        if metadata.access_level == ToolAccessLevel.CONDITIONAL:
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                return False
            return metadata.category in profile.allowed_categories

        return False

    async def create_agent_profile(
        self,
        agent_id: str,
        rag_enabled: bool = True,
        allowed_categories: Optional[Set[ToolCategory]] = None
    ) -> AgentToolProfile:
        """
        Create a tool profile for an agent - ENHANCED.

        Args:
            agent_id: Agent identifier
            rag_enabled: Whether agent has RAG access
            allowed_categories: Allowed tool categories

        Returns:
            Agent tool profile
        """
        try:
            if agent_id in self.agent_profiles:
                logger.warning(f"Tool profile already exists for agent {agent_id}")
                return self.agent_profiles[agent_id]

            # Create profile
            profile = AgentToolProfile.create(agent_id)
            self.agent_profiles[agent_id] = profile

            # Assign public tools by default
            for tool_id, metadata in self.tool_metadata.items():
                if metadata.access_level == ToolAccessLevel.PUBLIC:
                    profile.assigned_tools.add(tool_id)

            self.stats["total_agents"] += 1

            logger.info(f"Created tool profile for agent {agent_id}")
            return profile

        except Exception as e:
            logger.error(f"Failed to create tool profile for agent {agent_id}: {str(e)}")
            raise

    async def get_agent_tools(self, agent_id: str) -> List[BaseTool]:
        """
        Get all tools available to an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of available tools
        """
        try:
            if agent_id not in self.agent_profiles:
                await self.create_agent_profile(agent_id)

            profile = self.agent_profiles[agent_id]
            available_tools = []

            for tool_id in profile.assigned_tools:
                if tool_id in self.tools:
                    available_tools.append(self.tools[tool_id])

            return available_tools

        except Exception as e:
            logger.error(f"Failed to get tools for agent {agent_id}: {str(e)}")
            return []
    async def assign_tool_to_agent(
        self,
        agent_id: str,
        tool_id: str
    ) -> bool:
        """
        Assign a specific tool to an agent.

        Args:
            agent_id: Agent identifier
            tool_id: Tool identifier

        Returns:
            True if assignment successful
        """
        try:
            if tool_id not in self.tools:
                logger.error(f"Tool {tool_id} not found")
                return False

            if agent_id not in self.agent_profiles:
                await self.create_agent_profile(agent_id)

            profile = self.agent_profiles[agent_id]

            # Assign tool
            profile.assigned_tools.add(tool_id)
            profile.last_updated = datetime.now()

            logger.info(f"Assigned tool {tool_id} to agent {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to assign tool {tool_id} to agent {agent_id}: {str(e)}")
            return False
    async def record_tool_usage(
        self,
        agent_id: str,
        tool_id: str
    ) -> None:
        """
        Record tool usage for analytics.

        Args:
            agent_id: Agent that used the tool
            tool_id: Tool that was used
        """
        try:
            # Update agent profile
            if agent_id in self.agent_profiles:
                profile = self.agent_profiles[agent_id]
                if tool_id not in profile.usage_stats:
                    profile.usage_stats[tool_id] = 0
                profile.usage_stats[tool_id] += 1

            # Update tool metadata
            if tool_id in self.tool_metadata:
                metadata = self.tool_metadata[tool_id]
                metadata.usage_count += 1

            # Update global stats
            self.stats["total_tool_calls"] += 1

        except Exception as e:
            logger.error(f"Failed to record tool usage: {str(e)}")

    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Get a tool by ID, loading it on-demand if needed."""
        # Check if tool is already loaded
        if tool_id in self.tools:
            return self.tools[tool_id]

        # Try to load the tool on-demand
        try:
            import asyncio
            # Run the async loading in the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we can't use run_until_complete
                # So we'll need to handle this differently
                logger.warning(f"Tool {tool_id} not loaded yet - consider using async loading")
                return None
            else:
                # Load the tool synchronously
                loop.run_until_complete(self._load_tool_on_demand(tool_id))
                return self.tools.get(tool_id)
        except Exception as e:
            logger.error(f"Failed to load tool {tool_id} on-demand: {str(e)}")
            return None

    def get_tool_metadata(self, tool_id: str) -> Optional[ToolMetadata]:
        """Get tool metadata by ID."""
        return self.tool_metadata.get(tool_id)

    def get_agent_profile(self, agent_id: str) -> Optional[AgentToolProfile]:
        """Get agent tool profile."""
        return self.agent_profiles.get(agent_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get repository statistics."""
        return {
            **self.stats,
            "is_initialized": self.is_initialized,
            "tools_count": len(self.tools),
            "agent_profiles_count": len(self.agent_profiles)
        }

    # ================================
    # 🚀 ENHANCED TOOL LOADING METHODS
    # ================================

    async def _load_tools_for_new_use_case(self, use_case: str):
        """Load tools for a new use case dynamically."""
        try:
            # Map use cases to tools - ENHANCED MAPPING
            use_case_tool_mapping = {
                'stock_trading': ['advanced_stock_trading'],
                'financial_analysis': ['advanced_stock_trading', 'general_business_intelligence'],
                'business_analysis': ['general_business_intelligence'],
                'document_generation': ['revolutionary_file_generation'],
                'web_research': ['revolutionary_web_scraper'],
                'data_analysis': ['calculator'],
                'risk_management': ['advanced_stock_trading'],
                'trading': ['advanced_stock_trading'],
                'market_analysis': ['advanced_stock_trading'],
                'portfolio_management': ['advanced_stock_trading']
            }

            if use_case in use_case_tool_mapping:
                tool_names = use_case_tool_mapping[use_case]

                for tool_name in tool_names:
                    if tool_name == 'advanced_stock_trading':
                        await self._load_stock_trading_tool()
                    elif tool_name == 'revolutionary_file_generation':
                        await self._load_file_generation_tool()
                    # Add other tool loading logic here as needed

        except Exception as e:
            logger.error(f"Failed to load tools for use case {use_case}: {str(e)}")

    async def _load_tool_on_demand(self, tool_id: str) -> bool:
        """Load a specific tool on-demand."""
        try:
            if tool_id == 'advanced_stock_trading':
                await self._load_stock_trading_tool()
                return True
            elif tool_id == 'revolutionary_file_generation':
                await self._load_file_generation_tool()
                return True
            elif tool_id == 'screen_capture':
                await self._load_screen_capture_tool()
                return True
            elif tool_id == 'viral_content_generator':
                await self._load_viral_content_generator_tool()
                return True
            elif tool_id == 'ai_music_composition':
                await self._load_ai_music_composition_tool()
                return True
            elif tool_id == 'ai_lyric_vocal_synthesis':
                await self._load_ai_lyric_vocal_synthesis_tool()
                return True
            elif tool_id == 'meme_generation':
                await self._load_meme_generation_tool()
                return True
            elif tool_id == 'meme_collection':
                await self._load_meme_collection_tool()
                return True
            elif tool_id == 'meme_analysis':
                await self._load_meme_analysis_tool()
                return True
            elif tool_id == 'social_media_orchestrator':
                await self._load_social_media_orchestrator_tool()
                return True
            elif tool_id == 'revolutionary_web_scraper':
                await self._load_revolutionary_web_scraper_tool()
                return True
            elif tool_id == 'calculator':
                await self._load_calculator_tool()
                return True
            elif tool_id == 'text_processing_nlp':
                await self._load_text_processing_nlp_tool()
                return True
            elif tool_id == 'browser_automation':
                await self._load_browser_automation_tool()
                return True
            elif tool_id == 'notification_alert':
                await self._load_notification_alert_tool()
                return True
            else:
                logger.warning(f"No on-demand loader available for tool: {tool_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to load tool {tool_id} on-demand: {str(e)}")
            return False

    async def _load_stock_trading_tool(self):
        """Load the advanced stock trading tool."""
        try:
            from app.tools.production.advanced_stock_trading_tool import AdvancedStockTradingTool

            tool_instance = AdvancedStockTradingTool()

            metadata = ToolMetadata(
                tool_id="advanced_stock_trading",
                name="Advanced Stock Trading Tool",
                description="Comprehensive stock trading tool with real-time analysis and decision-making",
                category=ToolCategory.TRADING,
                access_level=ToolAccessLevel.PUBLIC,
                requires_rag=False,
                use_cases={
                    "stock_trading",
                    "financial_analysis",
                    "trading",
                    "market_analysis",
                    "portfolio_management",
                    "risk_management"
                }
            )

            await self.register_tool(tool_instance, metadata)
            logger.info("Advanced stock trading tool loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load stock trading tool: {str(e)}")

    async def _load_file_generation_tool(self):
        """Load the revolutionary file generation tool."""
        try:
            from app.tools.production.revolutionary_file_generation_tool import RevolutionaryFileGenerationTool

            tool_instance = RevolutionaryFileGenerationTool()

            metadata = ToolMetadata(
                tool_id="revolutionary_file_generation",
                name="Revolutionary File Generation Tool",
                description="The most powerful document generation system ever created - generates ANY type of file",
                category=ToolCategory.PRODUCTIVITY,
                access_level=ToolAccessLevel.PUBLIC,
                requires_rag=False,
                use_cases={
                    "document_generation",
                    "file_creation",
                    "report_generation",
                    "data_export",
                    "presentation_creation",
                    "spreadsheet_generation",
                    "visualization_creation",
                    "web_development",
                    "file_modification",
                    "batch_processing"
                }
            )

            await self.register_tool(tool_instance, metadata)
            logger.info("Revolutionary file generation tool loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load file generation tool: {str(e)}")

    async def _load_screen_capture_tool(self):
        """Load screen capture tool on-demand."""
        try:
            from app.tools.production.screen_capture_tool import get_screen_capture_tool, SCREEN_CAPTURE_TOOL_METADATA

            tool_instance = get_screen_capture_tool()

            # Register with metadata system
            from app.tools.metadata import get_global_registry
            registry = get_global_registry()
            registry.register_tool(tool_instance)

            await self.register_tool(tool_instance, SCREEN_CAPTURE_TOOL_METADATA)
            logger.info("✅ Screen capture tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load screen capture tool: {str(e)}")

    async def _load_viral_content_generator_tool(self):
        """Load viral content generator tool on-demand."""
        try:
            from app.tools.social_media.viral_content_generator_tool import get_viral_content_generator_tool, VIRAL_CONTENT_GENERATOR_TOOL_METADATA

            tool_instance = get_viral_content_generator_tool()

            # Register with metadata system
            from app.tools.metadata import get_global_registry
            registry = get_global_registry()
            registry.register_tool(tool_instance)

            await self.register_tool(tool_instance, VIRAL_CONTENT_GENERATOR_TOOL_METADATA)
            logger.info("✅ Viral content generator tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load viral content generator tool: {str(e)}")

    async def _load_ai_music_composition_tool(self):
        """Load AI music composition tool on-demand."""
        try:
            from app.tools.production.ai_music_composition_tool import get_ai_music_composition_tool, AI_MUSIC_COMPOSITION_TOOL_METADATA

            tool_instance = get_ai_music_composition_tool()

            # Register with metadata system
            from app.tools.metadata import get_global_registry
            registry = get_global_registry()
            registry.register_tool(tool_instance)

            await self.register_tool(tool_instance, AI_MUSIC_COMPOSITION_TOOL_METADATA)
            logger.info("✅ AI music composition tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load AI music composition tool: {str(e)}")

    async def _load_ai_lyric_vocal_synthesis_tool(self):
        """Load AI lyric vocal synthesis tool on-demand."""
        try:
            from app.tools.production.ai_lyric_vocal_synthesis_tool import get_ai_lyric_vocal_synthesis_tool

            tool_instance = get_ai_lyric_vocal_synthesis_tool()

            # Register with metadata system
            from app.tools.metadata import get_global_registry
            registry = get_global_registry()
            registry.register_tool(tool_instance)

            await self.register_tool(tool_instance, None)  # Metadata handled by registry
            logger.info("✅ AI lyric vocal synthesis tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load AI lyric vocal synthesis tool: {str(e)}")

    # ================================
    # 🚀 REVOLUTIONARY AUTO-DISCOVERY INTEGRATION
    # ================================

    async def auto_discover_and_register_tools(self) -> Dict[str, Any]:
        """
        Automatically discover and register all tools using the revolutionary auto-discovery system.

        Returns:
            Comprehensive discovery and registration report
        """
        try:
            logger.info("🔍 Starting revolutionary auto-discovery and registration...")

            # Import auto-discovery systems
            from app.tools.auto_discovery.tool_scanner import ToolAutoDiscovery
            from app.tools.auto_discovery.enhanced_registration import EnhancedRegistrationSystem

            # Initialize auto-discovery
            auto_discovery = ToolAutoDiscovery(self)

            # Discover all tools
            discovered_tools = await auto_discovery.discover_all_tools()
            logger.info(f"🔍 Discovered {len(discovered_tools)} tools")

            # Initialize enhanced registration
            registration_system = EnhancedRegistrationSystem(self)

            # Register discovered tools
            validated_tools = [
                tool_info for tool_info in discovered_tools.values()
                if tool_info.status.value in ['validated', 'discovered']
            ]

            registration_results = await registration_system.register_tools_batch(validated_tools)

            # Generate comprehensive report
            discovery_report = auto_discovery.generate_discovery_report()
            registration_report = registration_system.get_registration_report()

            combined_report = {
                "auto_discovery_enabled": True,
                "discovery_results": discovery_report,
                "registration_results": registration_report,
                "total_tools_discovered": len(discovered_tools),
                "total_tools_registered": len([r for r in registration_results.values() if r.value == 'registered']),
                "system_health": await self._generate_system_health_report(),
                "recommendations": self._generate_combined_recommendations(discovery_report, registration_report)
            }

            logger.info("🚀 Revolutionary auto-discovery and registration complete!")
            return combined_report

        except Exception as e:
            logger.error(f"Auto-discovery and registration failed: {str(e)}")
            return {
                "auto_discovery_enabled": False,
                "error": str(e),
                "fallback_mode": True
            }

    async def test_all_registered_tools(self) -> Dict[str, Any]:
        """
        Test all registered tools using the universal testing framework.

        Returns:
            Comprehensive testing report
        """
        try:
            logger.info("🧪 Starting comprehensive tool testing...")

            from app.tools.testing.universal_tool_tester import UniversalToolTester

            tester = UniversalToolTester()
            test_results = {}

            for tool_id, tool_instance in self.tools.items():
                try:
                    logger.info(f"🧪 Testing tool: {tool_id}")
                    test_result = await tester.test_tool_comprehensive(tool_instance)
                    test_results[tool_id] = {
                        "overall_success": test_result.overall_success,
                        "quality_score": test_result.quality_score,
                        "issues_count": len(test_result.issues),
                        "critical_issues": len([i for i in test_result.issues if i.severity.value == 'critical']),
                        "execution_time": test_result.metrics.execution_time,
                        "memory_usage": test_result.metrics.memory_usage,
                        "recommendations": test_result.recommendations
                    }
                except Exception as e:
                    test_results[tool_id] = {
                        "overall_success": False,
                        "error": str(e),
                        "quality_score": 0.0
                    }

            # Generate summary
            successful_tests = sum(1 for r in test_results.values() if r.get("overall_success", False))
            avg_quality_score = sum(r.get("quality_score", 0) for r in test_results.values()) / len(test_results) if test_results else 0

            testing_report = {
                "testing_timestamp": datetime.utcnow().isoformat(),
                "total_tools_tested": len(test_results),
                "successful_tests": successful_tests,
                "failed_tests": len(test_results) - successful_tests,
                "average_quality_score": avg_quality_score,
                "test_results": test_results,
                "system_recommendations": self._generate_testing_recommendations(test_results)
            }

            logger.info(f"🧪 Testing complete: {successful_tests}/{len(test_results)} tools passed")
            return testing_report

        except Exception as e:
            logger.error(f"Tool testing failed: {str(e)}")
            return {
                "testing_enabled": False,
                "error": str(e)
            }

    async def _generate_system_health_report(self) -> Dict[str, Any]:
        """Generate system health report."""
        return {
            "total_tools": len(self.tools),
            "total_metadata": len(self.tool_metadata),
            "agent_profiles": len(self.agent_profiles),
            "use_cases_mapped": len(self.use_case_tools),
            "repository_initialized": self.is_initialized,
            "stats": self.stats
        }

    def _generate_combined_recommendations(self, discovery_report: Dict, registration_report: Dict) -> List[str]:
        """Generate combined recommendations from discovery and registration reports."""
        recommendations = []

        # Add discovery recommendations
        if "recommendations" in discovery_report:
            recommendations.extend(discovery_report["recommendations"])

        # Add registration recommendations
        if "recommendations" in registration_report:
            recommendations.extend(registration_report["recommendations"])

        # Add system-level recommendations
        if len(self.tools) < 10:
            recommendations.append("🔧 Consider adding more tools to expand system capabilities")

        if self.stats["rag_enabled_tools"] == 0:
            recommendations.append("🧠 Consider adding RAG-enabled tools for enhanced knowledge capabilities")

        return list(set(recommendations))  # Remove duplicates

    def _generate_testing_recommendations(self, test_results: Dict) -> List[str]:
        """Generate recommendations based on testing results."""
        recommendations = []

        failed_tools = [tool_id for tool_id, result in test_results.items() if not result.get("overall_success", False)]
        if failed_tools:
            recommendations.append(f"🔧 Fix {len(failed_tools)} failing tools: {', '.join(failed_tools[:5])}")

        low_quality_tools = [
            tool_id for tool_id, result in test_results.items()
            if result.get("quality_score", 0) < 50
        ]
        if low_quality_tools:
            recommendations.append(f"⚡ Improve {len(low_quality_tools)} low-quality tools")

        slow_tools = [
            tool_id for tool_id, result in test_results.items()
            if result.get("execution_time", 0) > 5.0
        ]
        if slow_tools:
            recommendations.append(f"🚀 Optimize {len(slow_tools)} slow-performing tools")

        return recommendations

    async def _load_meme_generation_tool(self):
        """Load meme generation tool on-demand."""
        try:
            from app.tools.meme_generation_tool import meme_generation_tool, MEME_GENERATION_TOOL_METADATA

            # Register with metadata system
            from app.tools.metadata import get_global_registry
            registry = get_global_registry()
            registry.register_tool(meme_generation_tool)

            await self.register_tool(meme_generation_tool, MEME_GENERATION_TOOL_METADATA)
            logger.info("✅ Meme generation tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load meme generation tool: {str(e)}")

    async def _load_meme_collection_tool(self):
        """Load meme collection tool on-demand."""
        try:
            from app.tools.meme_collection_tool import meme_collection_tool, MEME_COLLECTION_TOOL_METADATA
            from app.tools.metadata import get_global_registry

            # Register with metadata system
            registry = get_global_registry()
            registry.register_tool(meme_collection_tool)

            await self.register_tool(meme_collection_tool, MEME_COLLECTION_TOOL_METADATA)
            logger.info("✅ Meme collection tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load meme collection tool: {str(e)}")

    async def _load_meme_analysis_tool(self):
        """Load meme analysis tool on-demand."""
        try:
            from app.tools.meme_analysis_tool import meme_analysis_tool, MEME_ANALYSIS_TOOL_METADATA
            from app.tools.metadata import get_global_registry

            # Register with metadata system
            registry = get_global_registry()
            registry.register_tool(meme_analysis_tool)

            await self.register_tool(meme_analysis_tool, MEME_ANALYSIS_TOOL_METADATA)
            logger.info("✅ Meme analysis tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load meme analysis tool: {str(e)}")

    async def _load_social_media_orchestrator_tool(self):
        """Load social media orchestrator tool on-demand."""
        try:
            from app.tools.social_media.social_media_orchestrator_tool import get_social_media_orchestrator_tool, SOCIAL_MEDIA_ORCHESTRATOR_TOOL_METADATA

            tool_instance = get_social_media_orchestrator_tool()

            # Register with metadata system
            from app.tools.metadata import get_global_registry
            registry = get_global_registry()
            registry.register_tool(tool_instance)

            await self.register_tool(tool_instance, SOCIAL_MEDIA_ORCHESTRATOR_TOOL_METADATA)
            logger.info("✅ Social media orchestrator tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load social media orchestrator tool: {str(e)}")

    async def _load_revolutionary_web_scraper_tool(self):
        """Load revolutionary web scraper tool on-demand."""
        try:
            from app.tools.production.revolutionary_web_scraper_tool import get_revolutionary_web_scraper_tool
            from app.tools.metadata import get_global_registry

            tool_instance = get_revolutionary_web_scraper_tool()

            # Register with metadata system
            registry = get_global_registry()
            registry.register_tool(tool_instance)

            await self.register_tool(tool_instance, None)  # Metadata handled by registry
            logger.info("✅ Revolutionary web scraper tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load revolutionary web scraper tool: {str(e)}")

    async def _load_calculator_tool(self):
        """Load calculator tool on-demand."""
        try:
            from app.tools.calculator_tool import calculator_tool, CALCULATOR_TOOL_METADATA
            from app.tools.metadata import get_global_registry

            # Register with metadata system
            registry = get_global_registry()
            registry.register_tool(calculator_tool)

            await self.register_tool(calculator_tool, CALCULATOR_TOOL_METADATA)
            logger.info("✅ Calculator tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load calculator tool: {str(e)}")

    async def _load_text_processing_nlp_tool(self):
        """Load text processing NLP tool on-demand."""
        try:
            from app.tools.production.text_processing_nlp_tool import get_text_processing_nlp_tool, TEXT_PROCESSING_NLP_TOOL_METADATA

            tool_instance = get_text_processing_nlp_tool()

            # Register with metadata system
            from app.tools.metadata import get_global_registry
            registry = get_global_registry()
            registry.register_tool(tool_instance)

            await self.register_tool(tool_instance, TEXT_PROCESSING_NLP_TOOL_METADATA)
            logger.info("✅ Text processing NLP tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load text processing NLP tool: {str(e)}")

    async def _load_browser_automation_tool(self):
        """Load browser automation tool on-demand."""
        try:
            from app.tools.production.browser_automation_tool import get_browser_automation_tool, BROWSER_AUTOMATION_TOOL_METADATA
            from app.tools.metadata import get_global_registry

            tool_instance = get_browser_automation_tool()

            # Register with metadata system
            registry = get_global_registry()
            registry.register_tool(tool_instance)

            await self.register_tool(tool_instance, BROWSER_AUTOMATION_TOOL_METADATA)
            logger.info("✅ Browser automation tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load browser automation tool: {str(e)}")

    async def _load_notification_alert_tool(self):
        """Load notification alert tool on-demand."""
        try:
            from app.tools.production.notification_alert_tool import notification_alert_tool, NOTIFICATION_ALERT_TOOL_METADATA
            from app.tools.metadata import get_global_registry

            # Register with metadata system
            registry = get_global_registry()
            registry.register_tool(notification_alert_tool)

            await self.register_tool(notification_alert_tool, NOTIFICATION_ALERT_TOOL_METADATA)
            logger.info("✅ Notification alert tool loaded on-demand")
        except Exception as e:
            logger.error(f"Failed to load notification alert tool: {str(e)}")


# Global instance
_unified_tool_repository: Optional[UnifiedToolRepository] = None


def get_unified_tool_repository() -> Optional[UnifiedToolRepository]:
    """Get the global unified tool repository instance."""
    global _unified_tool_repository

    if _unified_tool_repository is None:
        try:
            from app.core.unified_system_orchestrator import get_enhanced_system_orchestrator
            orchestrator = get_enhanced_system_orchestrator()

            if orchestrator and hasattr(orchestrator, 'tool_repository'):
                _unified_tool_repository = orchestrator.tool_repository

        except Exception as e:
            logger.error(f"Failed to get unified tool repository: {e}")

    return _unified_tool_repository



